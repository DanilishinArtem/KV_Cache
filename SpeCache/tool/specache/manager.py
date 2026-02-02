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
    def __init__(self, head_dim: int, num_kv_heads: int, num_q_heads: int, device: torch.device, 
                 bits: int = 8, max_cache_len: int = 32768):
        self.timer = Timer()
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.num_q_heads = num_q_heads
        self.device = device
        self.bits = bits
        self.qmin, self.qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        
        # 1. GPU Индекс (C') - фиксированный буфер для скорости
        self.k_quant_gpu = torch.zeros((num_kv_heads, max_cache_len, head_dim), device=device, dtype=torch.int8)
        self.k_scales_gpu = torch.zeros((num_kv_heads, max_cache_len, 1), device=device, dtype=torch.float32)
        self.v_quant_gpu = torch.zeros((num_kv_heads, max_cache_len, head_dim), device=device, dtype=torch.int8)
        self.v_scales_gpu = torch.zeros((num_kv_heads, max_cache_len, 1), device=device, dtype=torch.float32)
        
        # 2. CPU Хранилище (C) - ПРЕДВЫДЕЛЕННОЕ (pin_memory для скорости передачи)
        self.k_cpu = torch.zeros((num_kv_heads, max_cache_len, head_dim), pin_memory=True)
        self.v_cpu = torch.zeros((num_kv_heads, max_cache_len, head_dim), pin_memory=True)
        
        self.curr_ptr = 0 # Указатель на текущую позицию в кэше
        self.prefetched_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None 

    def _quantize(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Per-head quantization (dim 0: heads)
        max_vals = t.abs().amax(dim=(1, 2), keepdim=True)
        scales = (max_vals / self.qmax).clamp(min=1e-9)
        q = (t / scales).round().clamp(self.qmin, self.qmax).to(torch.int8)
        return q, scales

    def offload_and_update_index(self, K: torch.Tensor, V: torch.Tensor):
        self.timer.start("offload_and_update_index")
        """Оптимизированная запись без torch.cat (Zero-copy update)"""
        k_flat, v_flat = K.squeeze(0), V.squeeze(0)
        seq_len = k_flat.shape[1]
        end_ptr = self.curr_ptr + seq_len

        # Квантуем
        q_k, s_k = self._quantize(k_flat)
        q_v, s_v = self._quantize(v_flat)

        # Пишем в GPU индекс (для скоринга)
        self.k_quant_gpu[:, self.curr_ptr:end_ptr] = q_k
        self.k_scales_gpu[:, self.curr_ptr:end_ptr] = s_k.expand(-1, seq_len, -1)
        self.v_quant_gpu[:, self.curr_ptr:end_ptr] = q_v
        self.v_scales_gpu[:, self.curr_ptr:end_ptr] = s_v.expand(-1, seq_len, -1)

        # Пишем в CPU хранилище (Copy Host to Host - очень быстро)
        self.k_cpu[:, self.curr_ptr:end_ptr].copy_(k_flat, non_blocking=True)
        self.v_cpu[:, self.curr_ptr:end_ptr].copy_(v_flat, non_blocking=True)

        self.curr_ptr = end_ptr
        self.timer.end()

    def score_quantized(self, Q: torch.Tensor) -> torch.Tensor:
        self.timer.start("score_quantized")
        """Реализует быстрый поиск по C' на GPU"""
        if self.curr_ptr == 0: return None
        
        # Берем только заполненную часть кэша
        K_q = self.k_quant_gpu[:, :self.curr_ptr]
        S_k = self.k_scales_gpu[:, :self.curr_ptr]
        
        # Деквантуем 'на лету' (в float16/bf16 для скорости, если модель в них)
        K_deq = K_q.to(Q.dtype) * S_k.to(Q.dtype)
        
        # GQA Broadcast
        if self.num_q_heads != self.num_kv_heads:
            num_groups = self.num_q_heads // self.num_kv_heads
            K_deq = K_deq.repeat_interleave(num_groups, dim=0)

        # Matmul: (h, 1, hd) @ (h, hd, seq) -> (h, 1, seq)
        scores = torch.matmul(Q.squeeze(0), K_deq.transpose(-1, -2))
        self.timer.end()
        return scores.mean(dim=(0, 1)) # Aggregated importance

    def prefetch_next_step(self, scores: torch.Tensor, topk: int):
        self.timer.start("prefetch_next_step")
        """Асинхронный префетч (K_{t+1})"""
        if scores is None: return
        
        k = min(topk, self.curr_ptr)
        _, idxs = torch.topk(scores, k=k, dim=-1)
        unique_indices = idxs.flatten().unique().cpu()

        # Gather на CPU + асинхронный трансфер на GPU
        # По статье SpeCache: это скрывает задержку PCIe
        k_batch = self.k_cpu[:, unique_indices].to(self.device, non_blocking=True)
        v_batch = self.v_cpu[:, unique_indices].to(self.device, non_blocking=True)
        
        self.prefetched_kv = (k_batch.unsqueeze(0), v_batch.unsqueeze(0), unique_indices)
        self.timer.end()

    def get_full_kv(self, current_k: torch.Tensor, current_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.timer.start("get_full_kv")
        """
        Реализует K = [K' ∪ K_{Kt}, Kt]
        K' - весь квантованный кэш с GPU
        K_{Kt} - точные значения, подтянутые с CPU (заменяют квантованные в нужных индексах)
        """
        if self.curr_ptr == 0:
            return current_k, current_v

        # 1. Деквантуем ВЕСЬ кэш на GPU (K' и V')
        # По статье V тоже может быть квантован для этого шага
        # (Если вы не храните V_quant, придется подтягивать всё из CPU, но это медленно.
        #  SpeCache подразумевает, что на GPU лежит квантованный K' и V')
        k_full = self.k_quant_gpu[:, :self.curr_ptr].to(current_k.dtype) * self.k_scales_gpu[:, :self.curr_ptr]
        v_full = self.v_quant_gpu[:, :self.curr_ptr].to(current_v.dtype) * self.v_scales_gpu[:, :self.curr_ptr]

        # 2. Заменяем Top-K "заплатки" на точные данные с CPU
        if self.prefetched_kv is not None:
            pk, pv, p_indices = self.prefetched_kv # p_indices - индексы, которые мы тянули
            
            # Заменяем в деквантованном кэше ленивые значения на точные
            k_full[:, p_indices] = pk.squeeze(0)
            v_full[:, p_indices] = pv.squeeze(0)

        # 3. Добавляем текущий токен Kt
        # k_full: (h, seq_old, hd), current_k: (1, h, 1, hd)
        k_final = torch.cat([k_full.unsqueeze(0), current_k], dim=2)
        v_final = torch.cat([v_full.unsqueeze(0), current_v], dim=2)

        self.timer.end()

        return k_final, v_final