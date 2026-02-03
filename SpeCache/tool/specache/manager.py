import torch
from typing import Dict, List, Optional, Tuple
import time

class Timer:
    def __init__(self):
        self.info = False
        self._start = None
        self._end = None
        self._name = None
    def start(self, name=""):
        self._name = name
        self._start = time.time()
    def end(self):
        self._end = time.time()
        if self.info:
            print(f"[{self._name}] Time taken: {self._end - self._start}")


class SpeCacheManager:
    """
    Faithful SpeCache implementation (paper-aligned).
    """

    def __init__(
        self,
        num_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        device: torch.device,
        dtype=torch.float16,
        prefetch_k: int = 64,
    ):
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.prefetch_k = prefetch_k
        self.device = device
        self.dtype = dtype

        # ==============================
        # GPU: quantized keys
        # ==============================
        self.k_q = torch.empty(
            (num_kv_heads, max_seq_len, head_dim),
            dtype=torch.int8,
            device=device,
        )

        self.k_scale = torch.empty(
            (num_kv_heads, max_seq_len, 1),
            dtype=torch.float16,
            device=device,
        )

        # ==============================
        # GPU: exact values
        # ==============================
        self.v_gpu = torch.empty(
            (num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )

        # ==============================
        # CPU: exact keys (pinned)
        # ==============================
        self.k_cpu = torch.empty(
            (num_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )

        self.curr_len = 0

        # Prefetched exact keys (GPU)
        self.prefetched_k = None
        self.prefetched_idx = None

    # ============================================================
    # Quantization (per-token, per-head)
    # ============================================================
    @torch.no_grad()
    def _quantize(self, k: torch.Tensor):
        """
        k: (H, T, D)
        """
        scale = k.abs().amax(dim=-1, keepdim=True) / 127.0
        scale.clamp_(min=1e-6)
        q = torch.round(k / scale).to(torch.int8)
        return q, scale

    # ============================================================
    # Append KV (called every step)
    # ============================================================
    @torch.no_grad()
    def append(self, k: torch.Tensor, v: torch.Tensor):
        """
        k, v: (H, 1, D)
        """
        t = self.curr_len

        q, s = self._quantize(k)

        self.k_q[:, t:t+1] = q
        self.k_scale[:, t:t+1] = s
        self.v_gpu[:, t:t+1] = v

        # exact key → CPU
        self.k_cpu[:, t:t+1].copy_(k.cpu(), non_blocking=True)

        self.curr_len += 1

    # ============================================================
    # INT8 scoring (NO full dequant!)
    # ============================================================
    def score(self, q: torch.Tensor):
        """
        q: (num_q_heads, 1, D)

        returns:
          importance scores per token: (T,)
        """
        # map Q-heads → KV-heads (GQA)
        if q.shape[0] != self.num_kv_heads:
            repeat = q.shape[0] // self.num_kv_heads
            q = q.view(self.num_kv_heads, repeat, 1, self.head_dim)
            q = q.mean(dim=1)

        # q: (H, 1, D)
        q = q.squeeze(1)  # (H, D)

        # INT8 dot-product approximation:
        # score ≈ sum(q * (k_q * scale))
        k_q = self.k_q[:, :self.curr_len]           # (H, T, D)
        s = self.k_scale[:, :self.curr_len]         # (H, T, 1)

        # matmul in INT space
        # (H, D) x (H, T, D) → (H, T)
        scores = torch.einsum("hd,htd->ht", q, k_q.float())
        scores = scores * s.squeeze(-1)

        # aggregate heads → token importance
        importance = scores.max(dim=0).values  # (T,)

        return importance

    # ============================================================
    # Prefetch exact keys (async)
    # ============================================================
    @torch.no_grad()
    def prefetch(self, importance: torch.Tensor):
        """
        importance: (T,)
        """
        k = min(self.prefetch_k, importance.numel())
        topk = torch.topk(importance, k=k, largest=True)

        idx = topk.indices.sort().values
        self.prefetched_idx = idx

        # async CPU → GPU copy
        self.prefetched_k = self.k_cpu[:, idx].to(
            device=self.device,
            non_blocking=True,
        )

    # ============================================================
    # Assemble attention KV
    # ============================================================
    def get_attention_kv(self, k_curr: torch.Tensor, v_curr: torch.Tensor):
        """
        returns:
          K_attn, V_attn
        """
        if self.prefetched_k is None:
            # fallback: only current token
            return k_curr, v_curr

        K = torch.cat([self.prefetched_k, k_curr], dim=1)
        V = torch.cat([self.v_gpu[:, self.prefetched_idx], v_curr], dim=1)

        return K, V

    # ============================================================
    # Reset (for new sequence)
    # ============================================================
    def reset(self):
        self.curr_len = 0
        self.prefetched_k = None
        self.prefetched_idx = None