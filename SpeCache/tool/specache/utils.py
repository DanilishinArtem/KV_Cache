import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Callable
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, eager_attention_forward
from transformers.utils.generic import TransformersKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers import AutoModelForCausalLM, AutoTokenizer
from specache.manager import SpeCacheManager
import time


def load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    return tokenizer, model


def Quen2Attention_init(self, config, layer_idx, specache_config):
    print("[INFO] Start of overriting initialization of qwen2 attention")
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True
    self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
    self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
    self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=True)
    self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
    self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
    # additional logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.bits = specache_config.get("bits", 8)
    self.topk = specache_config.get("topk", 64)
    self.spec_manager = SpeCacheManager(head_dim=self.head_dim, num_kv_heads=config.num_key_value_heads, num_q_heads=config.num_attention_heads, device=device, bits=self.bits)


def SpeCache_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    
    input_shape = hidden_states.shape[:-1]
    # На этапе генерации seq_len всегда 1 (если не используется спекулятивное декодирование)
    seq_len = hidden_states.shape[1] 
    
    # 1. Проекции
    query_states = self.q_proj(hidden_states).view(*input_shape, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(*input_shape, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(*input_shape, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
    
    # 2. RoPE
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    is_prefill = seq_len > 1

    if is_prefill:
        # print(f'[INFO] Prefill Stage (seq_len = {seq_len})')
        # Prefill Stage (согласно статье)
        # Вычисляем аттеншн по полной последовательности
        key_for_attn, value_for_attn = key_states, value_states
        
        # Сохраняем всё в SpeCache (Offload C)
        self.spec_manager.offload_and_update_index(key_states, value_states)
        # (Опционально) Обновляем стандартный кэш, если он нужен выше по коду
        if past_key_values is not None:
            past_key_values.update(key_states, value_states, self.layer_idx, {"sin": sin, "cos": cos, "cache_position": cache_position})

    else:
        # print(f'[INFO] Decoding Stage (seq_len = {seq_len})')
        # Decoding Stage (seq_len = 1)
        # 1. Скорим квантованный индекс на GPU (Look-ahead или текущий)
        query_states = query_states.view(*input_shape, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        scores = self.spec_manager.score_quantized(query_states)
        # 2. Выбираем индексы и подтягиваем данные (Prefetch C_{Kt})
        # Примечание: В идеале это должно было случиться в конце предыдущего шага
        self.spec_manager.prefetch_next_step(scores, self.topk)
        # 3. Собираем KV для аттеншена (K = [K_prefetch ∪ K_t])
        key_for_attn, value_for_attn = self.spec_manager.get_full_kv(key_states, value_states)
        # 4. Обновляем индекс оффлоада текущим токеном
        self.spec_manager.offload_and_update_index(key_states, value_states)
    # 3. Настройка маски
    # Т.к. SpeCache выбирает разреженные токены, стандартная казуальная маска не подходит.
    # Для seq_len=1 нам нужна маска, разрешающая смотреть на все префетченные токены + текущий.
    if not is_prefill:
        # Для eager_attention маска (bsz, 1, q_len, kv_len)
        kv_seq_len = key_for_attn.shape[-2]
        attention_mask = torch.zeros(
            (query_states.shape[0], 1, seq_len, kv_seq_len), 
            device=query_states.device, dtype=query_states.dtype
        )

    # 4. Вычисление Attention
    attn_output, attn_weights = eager_attention_forward(
        self,
        query_states,
        key_for_attn,
        value_for_attn,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
    )

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights


def CausalLM_forward(self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )