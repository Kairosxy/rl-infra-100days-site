"""Estimate naive Attention storage and optimizer state size on CPU.

This is a mathematical budget calculator, not a profiler.
Run: python examples/day_02/memory_budget.py
"""
from __future__ import annotations


def estimate_attention_score_bytes(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    bytes_per_element: int = 2,
) -> int:
    """Return bytes for a materialized [B, heads, L, L] tensor."""
    if min(batch_size, num_heads, seq_len, bytes_per_element) <= 0:
        raise ValueError("All shape and dtype arguments must be positive.")
    return batch_size * num_heads * seq_len**2 * bytes_per_element


def estimate_adamw_state_bytes(
    num_parameters: int,
    parameter_bytes: int = 2,
    gradient_bytes: int = 2,
    moment_bytes: int = 4,
    master_weight_bytes: int = 0,
) -> dict[str, int]:
    """Return unsharded state sizes, excluding activations and buffers."""
    if num_parameters <= 0 or any(
        n < 0
        for n in [parameter_bytes, gradient_bytes, moment_bytes, master_weight_bytes]
    ):
        raise ValueError("State sizes must be nonnegative; parameter count positive.")
    result = {
        "parameters": num_parameters * parameter_bytes,
        "gradients": num_parameters * gradient_bytes,
        "adam_m": num_parameters * moment_bytes,
        "adam_v": num_parameters * moment_bytes,
        "master_weights": num_parameters * master_weight_bytes,
    }
    result["total"] = sum(result.values())
    return result


def main() -> None:
    example = estimate_attention_score_bytes(2, 16, 640)
    large = estimate_attention_score_bytes(2, 16, 8320)
    budget = estimate_adamw_state_bytes(1_000_000_000)
    print(f"Naive Attention score @ L=640:  {example / 2**20:.2f} MiB")
    print(f"Naive Attention score @ L=8320: {large / 2**30:.2f} GiB")
    print(f"Length scaling: {large / example:.1f}x")
    print(f"1B AdamW state (assumed dtypes): {budget['total'] / 10**9:.1f} GB")
    print("Not a measured GPU peak: activation buffers and allocator not included.")


if __name__ == "__main__":
    main()
