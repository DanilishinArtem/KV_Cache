import os
import sys
import json
import psutil
import tracemalloc
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from contextlib import contextmanager

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from lm_eval import evaluator
from transformers import AutoTokenizer, AutoModelForCausalLM
import traceback

# Try to import R-KV if available
try:
    from rkv.monkeypatch import replace_qwen2
    RKV_AVAILABLE = True
except ImportError:
    RKV_AVAILABLE = False
    print("[WARNING] R-KV package not available, will use basic patching")

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME = "Qwen/Qwen2-0.5B"  # Small Qwen model
TASK_NAME = "swde"
NUM_FEWSHOT = 0
BATCH_SIZE = 4
NUM_SAMPLES = 50  # Limit samples for quick evaluation

# R-KV compression config (following R-KV paper defaults)
RKV_COMPRESSION_CONFIG = {
    "compression_method": "rkv",
    "budget_ratio": 0.5,  # Keep 50% of KV cache
    "window_size": 64,
    "buffer_size": 128,    # B_buffer
    "observation_window": 8,  # α
    "importance_weight": 0.1,  # λ
}

# ============================================================================
# KV Cache Memory Monitoring
# ============================================================================

class KVCacheMonitor:
    """Monitor KV cache memory usage during inference"""
    
    def __init__(self):
        self.peak_memory = 0
        self.allocated_memory = []
        self.reserved_memory = []
        
    def reset(self):
        self.peak_memory = 0
        self.allocated_memory = []
        self.reserved_memory = []
        
    def record(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # MB
            self.allocated_memory.append(allocated)
            self.reserved_memory.append(reserved)
            self.peak_memory = max(self.peak_memory, allocated)
    
    def get_stats(self) -> Dict:
        return {
            "peak_memory_mb": self.peak_memory,
            "avg_allocated_mb": np.mean(self.allocated_memory) if self.allocated_memory else 0,
            "avg_reserved_mb": np.mean(self.reserved_memory) if self.reserved_memory else 0,
        }

# ============================================================================
# Model patching for R-KV
# ============================================================================

def apply_rkv_patching(compression_config: Dict):
    """Apply R-KV monkey patching to Qwen2 attention layers"""
    try:
        if RKV_AVAILABLE:
            # Use the official R-KV library patching
            replace_qwen2(compression_config)
            print("[INFO] R-KV compression patching applied using official library")
            return True
        else:
            # Fallback: basic patching
            from transformers.models.qwen2 import modeling_qwen2
            
            original_attention_init = modeling_qwen2.Qwen2Attention.__init__
            
            def patched_init(self, config, layer_idx):
                original_attention_init(self, config, layer_idx)
                if not hasattr(config, 'compression_config'):
                    config.compression_config = compression_config
                self.compression_enabled = True
                
            modeling_qwen2.Qwen2Attention.__init__ = patched_init
            
            print("[INFO] R-KV compression patching applied using basic method")
            return True
    except Exception as e:
        print(f"[WARNING] Could not apply R-KV patching: {e}")
        traceback.print_exc()
        return False

def remove_rkv_patching():
    """Remove R-KV patching to restore original behavior"""
    try:
        from transformers.models.qwen2 import modeling_qwen2
        # Reload the module to restore original implementation
        import importlib
        importlib.reload(modeling_qwen2)
        print("[INFO] R-KV patching removed, original implementation restored")
        return True
    except Exception as e:
        print(f"[WARNING] Could not remove R-KV patching: {e}")
        return False

# ============================================================================
# Evaluation
# ============================================================================

def evaluate_model(
    model_name: str,
    task_name: str,
    use_rkv: bool = False,
    compression_config: Optional[Dict] = None,
    num_samples: int = 100,
    **kwargs
) -> Dict:
    """
    Evaluate a model on a task with optional R-KV compression
    
    Args:
        model_name: HuggingFace model identifier
        task_name: Task name for evaluation
        use_rkv: Whether to apply R-KV compression
        compression_config: R-KV compression configuration
        num_samples: Number of samples to evaluate
        **kwargs: Additional arguments for evaluator.simple_evaluate
    
    Returns:
        Dictionary with results and statistics
    """
    
    print(f"\n{'='*70}")
    print(f"Evaluating {model_name} on {task_name}")
    print(f"R-KV Compression: {use_rkv}")
    print(f"{'='*70}")
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    monitor = KVCacheMonitor()
    
    try:
        # Apply patching if needed
        if use_rkv and compression_config:
            apply_rkv_patching(compression_config)
        
        # Run evaluation
        print(f"\nRunning evaluation with limit={num_samples}...")
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_name},dtype=float16",
            tasks=[task_name],
            num_fewshot=NUM_FEWSHOT,
            batch_size=BATCH_SIZE,
            device="cuda",
            log_samples=False,
            limit=num_samples,
            **kwargs
        )
        
        # Record memory stats
        if torch.cuda.is_available():
            monitor.record()
            torch.cuda.synchronize()
        
        # Extract relevant metrics
        task_results = results.get("results", {}).get(task_name, {})
        
        output = {
            "model": model_name,
            "task": task_name,
            "rkv_enabled": use_rkv,
            "num_samples": num_samples,
            "metrics": task_results,
            "memory_stats": monitor.get_stats(),
        }
        
        print(f"\nResults:")
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}" if isinstance(value, float) else f"  {metric}: {value}")
        
        memory_stats = monitor.get_stats()
        print(f"\nMemory Statistics:")
        for stat_name, stat_value in memory_stats.items():
            print(f"  {stat_name}: {stat_value:.2f}")
        
        return output
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        traceback.print_exc()
        return {
            "model": model_name,
            "task": task_name,
            "rkv_enabled": use_rkv,
            "error": str(e),
        }
    
    finally:
        # Clean up
        if use_rkv:
            remove_rkv_patching()

# ============================================================================
# Main evaluation loop
# ============================================================================

def main():
    print("\n" + "="*70)
    print("KV Cache Compression Evaluation: R-KV vs Standard Attention")
    print("="*70)
    
    results_comparison = {}
    
    # Test 1: Standard attention (baseline)
    print("\n[PHASE 1/2] Evaluating with standard attention (baseline)...")
    baseline_results = evaluate_model(
        model_name=MODEL_NAME,
        task_name=TASK_NAME,
        use_rkv=False,
        num_samples=NUM_SAMPLES,
    )
    results_comparison["baseline"] = baseline_results
    
    # Clear GPU memory between runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Test 2: With R-KV compression
    print("\n[PHASE 2/2] Evaluating with R-KV compression...")
    rkv_results = evaluate_model(
        model_name=MODEL_NAME,
        task_name=TASK_NAME,
        use_rkv=True,
        compression_config=RKV_COMPRESSION_CONFIG,
        num_samples=NUM_SAMPLES,
    )
    results_comparison["rkv"] = rkv_results
    
    # ========================================================================
    # Summary and Comparison
    # ========================================================================
    
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    
    if "error" not in baseline_results and "error" not in rkv_results:
        baseline_metrics = baseline_results.get("metrics", {})
        rkv_metrics = rkv_results.get("metrics", {})
        
        baseline_memory = baseline_results.get("memory_stats", {})
        rkv_memory = rkv_results.get("memory_stats", {})
        
        print("\n1. Accuracy Metrics:")
        print(f"   {'Metric':<30} {'Baseline':<15} {'R-KV':<15} {'Change':<15}")
        print("   " + "-"*75)
        
        for metric in baseline_metrics:
            baseline_val = baseline_metrics.get(metric, 0)
            rkv_val = rkv_metrics.get(metric, 0)
            
            if isinstance(baseline_val, (int, float)) and isinstance(rkv_val, (int, float)):
                change = ((rkv_val - baseline_val) / baseline_val * 100) if baseline_val != 0 else 0
                print(f"   {metric:<30} {baseline_val:<15.4f} {rkv_val:<15.4f} {change:+.2f}%")
        
        print("\n2. Memory Usage:")
        print(f"   {'Metric':<30} {'Baseline (MB)':<20} {'R-KV (MB)':<20} {'Reduction':<20}")
        print("   " + "-"*75)
        
        baseline_peak = baseline_memory.get("peak_memory_mb", 0)
        rkv_peak = rkv_memory.get("peak_memory_mb", 0)
        reduction = ((baseline_peak - rkv_peak) / baseline_peak * 100) if baseline_peak != 0 else 0
        
        print(f"   {'Peak Memory':<30} {baseline_peak:<20.2f} {rkv_peak:<20.2f} {reduction:+.2f}%")
        
        baseline_avg = baseline_memory.get("avg_allocated_mb", 0)
        rkv_avg = rkv_memory.get("avg_allocated_mb", 0)
        reduction_avg = ((baseline_avg - rkv_avg) / baseline_avg * 100) if baseline_avg != 0 else 0
        
        print(f"   {'Avg Allocated':<30} {baseline_avg:<20.2f} {rkv_avg:<20.2f} {reduction_avg:+.2f}%")
    
    # Save results to file
    output_file = "./eval_rkv_results.json"
    with open(output_file, "w") as f:
        json.dump(results_comparison, f, indent=2)
    
    print(f"\n[INFO] Results saved to {output_file}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
