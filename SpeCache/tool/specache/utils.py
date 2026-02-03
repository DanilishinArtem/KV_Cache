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
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        seq_len = hidden_states.shape[1] 
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        is_prefill = seq_len > 1

        if is_prefill:
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

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_for_attn,
            value_for_attn,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights
