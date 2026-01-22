#!/usr/bin/env python3
"""
Pure KV Cache Compression Measurement using R-KV
==================================================

Generate exactly N tokens and measure GPU memory:
1. Baseline: Generate N tokens without compression
2. Compressed: Generate N tokens with R-KV compression (same seed)
3. Compare actual memory usage and savings

Uses the actual R-KV monkeypatch approach from HuggingFace integration.

Usage:
    python eval_pure_kv_compression.py --tokens 128 256 512
    python eval_pure_kv_compression.py --tokens 256 --seed 42 --method rkv
"""

import argparse
import json
import os
import sys
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add HuggingFace R-KV integration to path
RKVCACHE_HF_PATH = os.path.expanduser("~/work/kv_cache/R-KV/HuggingFace")
if os.path.exists(RKVCACHE_HF_PATH):
    sys.path.insert(0, RKVCACHE_HF_PATH)

try:
    from rkv.monkeypatch import replace_qwen2, replace_llama
    HAS_RKVCACHE = True
except ImportError as e:
    HAS_RKVCACHE = False
    print(f"[WARNING] R-KV HuggingFace integration not found: {e}")


class MemoryMonitor:
    """Track GPU memory usage during generation"""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.peak_memory = 0
        self.allocated_memory = 0
        
    def reset(self):
        """Reset tracking and clear cache"""
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        self.peak_memory = 0
        self.allocated_memory = 0
    
    def get_memory_mb(self) -> float:
        """Get current GPU memory usage in MB"""
        if self.device == "cuda":
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_peak_memory_mb(self) -> float:
        """Get peak GPU memory usage in MB"""
        if self.device == "cuda":
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0
    
    def record(self):
        """Record current memory state"""
        self.allocated_memory = self.get_memory_mb()
        self.peak_memory = self.get_peak_memory_mb()


def calculate_kv_cache_size_mb(
    batch_size: int,
    num_layers: int,
    seq_length: int,
    num_heads: int,
    head_dim: int,
    dtype: str = "float16"
) -> float:
    """
    Calculate theoretical KV cache size in MB.
    
    Formula: 2 * batch * layers * seq_len * heads * head_dim * bytes_per_value
    (Factor of 2 for K and V)
    """
    bytes_per_value = 2 if dtype == "float16" else 4  # float16=2 bytes, float32=4 bytes
    total_bytes = 2 * batch_size * num_layers * seq_length * num_heads * head_dim * bytes_per_value
    return total_bytes / (1024 * 1024)


def generate_tokens(
    model: nn.Module,
    tokenizer,
    prompt: str,
    num_tokens: int,
    seed: int = 42,
    device: str = "cuda"
) -> Tuple[str, int]:
    """
    Generate exactly num_tokens tokens from prompt.
    
    Args:
        model: Language model
        tokenizer: Tokenizer
        prompt: Input prompt
        num_tokens: Exact number of tokens to generate
        seed: Random seed for reproducibility
        device: Device to generate on
    
    Returns:
        (generated_text, actual_tokens_generated)
    """
    torch.manual_seed(seed)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=num_tokens,
            min_new_tokens=num_tokens,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    actual_tokens = output.shape[1] - input_length
    
    return generated_text, actual_tokens


def benchmark_compression(
    model_name: str = "Qwen/Qwen2-0.5B",
    token_counts: list = None,
    method: str = "rkv",
    kv_budget: int = 128,
    seed: int = 42,
    device: str = "cuda"
) -> dict:
    """
    Benchmark KV cache compression.
    
    Args:
        model_name: HuggingFace model ID
        token_counts: List of token counts to test
        method: Compression method (rkv, snapkv, streamingllm, h2o)
        kv_budget: KV cache budget for compression
        seed: Random seed
        device: Device to use
    
    Returns:
        Dictionary with results
    """
    if token_counts is None:
        token_counts = [128, 256, 512, 1024]
    
    print("=" * 80)
    print("PURE KV CACHE COMPRESSION MEASUREMENT (R-KV)")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Model: {model_name}")
    print(f"  Token counts: {token_counts}")
    print(f"  Compression method: {method}")
    print(f"  KV budget: {kv_budget}")
    print(f"  Seed: {seed}")
    print(f"  Device: {torch.cuda.get_device_name(0) if device == 'cuda' else device}")
    print(f"  R-KV available: {HAS_RKVCACHE}")
    print()
    
    # Load model and tokenizer
    print("[LOADING] Model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    
    # Get model config for KV calculation
    config = model.config
    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = config.hidden_size // config.num_attention_heads
    
    print(f"  Layers: {num_layers}, Heads: {num_heads}, Head dim: {head_dim}")
    print()
    
    # Prompt for generation
    prompt = "The quick brown fox jumped over the lazy dog. In response to this, I will generate a detailed explanation:"
    
    # Results storage
    results = {
        "model": model_name,
        "device": str(torch.cuda.get_device_name(0) if device == 'cuda' else device),
        "method": method,
        "kv_budget": kv_budget,
        "seed": seed,
        "model_config": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "hidden_size": config.hidden_size,
        },
        "benchmarks": []
    }
    
    memory_monitor = MemoryMonitor(device)
    
    # Test each token count
    for num_tokens in token_counts:
        print(f"Testing {num_tokens} tokens:")
        print("-" * 40)
        
        # BASELINE: Generate without compression
        print(f"  [1/2] Baseline (no compression)...", end=" ", flush=True)
        memory_monitor.reset()
        
        try:
            text_baseline, actual_tokens = generate_tokens(
                model, tokenizer, prompt, num_tokens, seed=seed, device=device
            )
            memory_monitor.record()
            peak_baseline = memory_monitor.get_peak_memory_mb()
            print(f"OK ({actual_tokens} tokens, {peak_baseline:.2f} MB)")
        except Exception as e:
            print(f"ERROR: {e}")
            peak_baseline = 0.0
            actual_tokens = 0
        
        # Calculate theoretical KV cache
        total_seq_len = len(tokenizer(prompt)["input_ids"]) + actual_tokens
        kv_size_theoretical = calculate_kv_cache_size_mb(
            batch_size=1,
            num_layers=num_layers,
            seq_length=total_seq_len,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype="float16"
        )
        
        # COMPRESSED: Generate with R-KV compression (if available)
        peak_compressed = 0.0
        savings_pct = 0.0
        savings_mb = 0.0
        
        if HAS_RKVCACHE:
            print(f"  [2/2] R-KV compressed...", end=" ", flush=True)
            
            try:
                # Reload model for fresh compression state
                model_compressed = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=device,
                )
                model_compressed.eval()
                
                # Build compression config using R-KV monkeypatch
                compression_config = {
                    "method": method,
                    "method_config": {
                        "budget": kv_budget,
                        "window_size": 8,
                        "mix_lambda": 0.07,
                        "retain_ratio": 0.2,
                        "retain_direction": "last",
                        "first_tokens": 4,
                    },
                    "compression": None,
                    "update_kv": True,
                }
                
                # Apply monkeypatch based on model type BEFORE creating the model
                if "qwen2" in model_name.lower():
                    replace_qwen2(compression_config)
                elif "qwen3" in model_name.lower():
                    from rkv.monkeypatch import replace_qwen3
                    replace_qwen3(compression_config)
                else:
                    replace_llama(compression_config)
                
                # Reload model AFTER monkeypatch to apply compression
                model_compressed = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=device,
                )
                model_compressed.eval()
                
                memory_monitor.reset()
                
                # Generate with same seed for fair comparison
                text_compressed, _ = generate_tokens(
                    model_compressed, tokenizer, prompt, num_tokens, seed=seed, device=device
                )
                memory_monitor.record()
                peak_compressed = memory_monitor.get_peak_memory_mb()
                
                savings_mb = peak_baseline - peak_compressed
                savings_pct = (savings_mb / peak_baseline * 100) if peak_baseline > 0 else 0.0
                
                print(f"OK ({peak_compressed:.2f} MB)")
                print(f"  Savings: {savings_mb:.2f} MB ({savings_pct:.1f}%)")
                
                # Clean up
                del model_compressed
                torch.cuda.empty_cache()
                
            except Exception as e:
                import traceback
                print(f"SKIP: {str(e)[:80]}")
                # traceback.print_exc()  # Uncomment for debugging
                peak_compressed = None
                savings_pct = None
        else:
            print(f"  [2/2] R-KV compressed...", end=" ", flush=True)
            print("SKIP (R-KV not installed)")
        
        # Store result
        result = {
            "num_tokens": num_tokens,
            "actual_tokens": actual_tokens,
            "peak_memory_baseline_mb": peak_baseline,
            "peak_memory_compressed_mb": peak_compressed,
            "savings_mb": savings_mb,
            "savings_pct": savings_pct,
            "theoretical_kv_cache_mb": kv_size_theoretical,
        }
        results["benchmarks"].append(result)
        print()
    
    return results


def print_summary_table(results: dict):
    """Print results as a formatted table"""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print()
    print(f"{'Tokens':<10} {'Peak Mem':<15} {'Compressed':<15} {'Savings':<15} {'Savings %':<10}")
    print("-" * 65)
    
    for bench in results["benchmarks"]:
        tokens = bench["num_tokens"]
        baseline = bench["peak_memory_baseline_mb"]
        compressed = bench["peak_memory_compressed_mb"]
        savings = bench["savings_mb"]
        savings_pct = bench["savings_pct"]
        
        baseline_str = f"{baseline:.1f} MB" if baseline else "N/A"
        compressed_str = f"{compressed:.1f} MB" if compressed else "N/A"
        savings_str = f"{savings:.1f} MB" if savings and savings > 0 else "N/A"
        pct_str = f"{savings_pct:.1f}%" if savings_pct else "N/A"
        
        print(f"{tokens:<10} {baseline_str:<15} {compressed_str:<15} {savings_str:<15} {pct_str:<10}")
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Measure pure KV cache compression using R-KV"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024],
        help="Token counts to test"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="Model name or path"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="rkv",
        choices=["rkv", "snapkv", "streamingllm", "h2o"],
        help="Compression method"
    )
    parser.add_argument(
        "--kv-budget",
        type=int,
        default=128,
        help="KV cache budget for compression"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="kv_compression_results.json",
        help="Output JSON file"
    )
    
    args = parser.parse_args()
    
    # Run benchmark
    results = benchmark_compression(
        model_name=args.model,
        token_counts=args.tokens,
        method=args.method,
        kv_budget=args.kv_budget,
        seed=args.seed,
        device=args.device
    )
    
    # Print summary
    print_summary_table(results)
    
    # Save results
    output_path = os.path.join("/home/adanilishin/work/kv_cache/logs", args.output)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")
