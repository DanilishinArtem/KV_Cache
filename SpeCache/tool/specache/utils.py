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
    # self.spec_manager = SpeCacheManager(head_dim=self.head_dim, num_kv_heads=config.num_key_value_heads, num_q_heads=config.num_attention_heads, device=device, bits=self.bits)
    self.spec_manager = SpeCacheManager(num_kv_heads=config.num_key_value_heads, head_dim=self.head_dim, max_seq_len=config.max_position_embeddings, device=device, dtype=torch.bfloat16, prefetch_k=self.topk)


def SpeCache_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, seq_len, _ = hidden_states.shape
        head_dim = self.head_dim
        num_heads = self.config.num_attention_heads

        # ============================================================
        # 1. QKV projection
        # ============================================================
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        q = q.view(bsz, seq_len, self.config.num_attention_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.config.num_key_value_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.config.num_key_value_heads, head_dim).transpose(1, 2)

        # ============================================================
        # 2. RoPE
        # ============================================================
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        is_prefill = seq_len > 1

        # ============================================================
        # 3. Prefill stage (FULL attention, FULL offload)
        # ============================================================
        if is_prefill:
            # используем обычный dense attention
            key_for_attn = k
            value_for_attn = v

            # весь KV уходит в SpeCache (CPU + quant index)
            self.spec_manager.reset()
            for t in range(seq_len):
                self.spec_manager.append(
                    k[:, :, t:t+1, :].squeeze(0),
                    v[:, :, t:t+1, :].squeeze(0),
                )

            # при необходимости обновляем HF cache
            if past_key_values is not None:
                past_key_values.update(
                    k, v, self.layer_idx,
                    {
                        "sin": sin,
                        "cos": cos,
                        "cache_position": cache_position,
                    },
                )

        # ============================================================
        # 4. Decode stage (SpeCache path)
        # ============================================================
        else:
            # q, k, v : (bsz=1, heads, 1, dim)
            q_t = q
            k_t = k
            v_t = v

            # 4.1 Используем PREFETCH из предыдущего шага
            key_for_attn, value_for_attn = self.spec_manager.get_attention_kv(
                k_t.squeeze(0),
                v_t.squeeze(0),
            )

            # приводим к HF-формату
            key_for_attn = key_for_attn.unsqueeze(0)
            value_for_attn = value_for_attn.unsqueeze(0)

            # 4.2 Скорим текущий Q → префетч ДЛЯ СЛЕДУЮЩЕГО шага
            importance = self.spec_manager.score(q_t.squeeze(0))
            self.spec_manager.prefetch(importance)

            # 4.3 Оффлоадим текущий токен
            self.spec_manager.append(
                k_t.squeeze(0),
                v_t.squeeze(0),
            )

            # маска: разрешаем смотреть на все выбранные токены
            kv_seq_len = key_for_attn.shape[-2]
            attention_mask = torch.zeros(
                (bsz, 1, 1, kv_seq_len),
                device=q.device,
                dtype=q.dtype,
            )

        # ============================================================
        # 5. Attention
        # ============================================================
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            q,
            key_for_attn,
            value_for_attn,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        # ============================================================
        # 6. Output projection
        # ============================================================
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
