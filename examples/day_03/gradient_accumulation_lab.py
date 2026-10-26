"""Compare full-batch gradients, accumulation, and premature optimizer steps.

Run: python examples/day_03/gradient_accumulation_lab.py
"""
from __future__ import annotations

import torch


def make_data() -> tuple[torch.Tensor, torch.Tensor]:
    """Provide deterministic CPU inputs with one uneven final micro-batch."""
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 2.0], [2.0, 1.0], [1.0, -1.0]],
        dtype=torch.float64,
    )
    targets = torch.tensor([[1.0], [0.0], [2.0], [1.0], [-1.0]], dtype=torch.float64)
    return inputs, targets


def make_model() -> torch.nn.Linear:
    model = torch.nn.Linear(2, 1, bias=False, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.3, -0.2]], dtype=torch.float64))
    return model


def compute_gradient(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    micro_batch_size: int | None = None,
    wrong_equal_microbatch_weighting: bool = False,
) -> torch.Tensor:
    """Compute MSE gradient; optionally demonstrate an invalid averaging rule."""
    model = make_model()
    model.zero_grad(set_to_none=True)
    if micro_batch_size is None:
        loss = (model(inputs) - targets).square().mean()
        loss.backward()
    else:
        if micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive.")
        batches = [
            (inputs[i : i + micro_batch_size], targets[i : i + micro_batch_size])
            for i in range(0, len(inputs), micro_batch_size)
        ]
        for micro_inputs, micro_targets in batches:
            squared_error = (model(micro_inputs) - micro_targets).square()
            if wrong_equal_microbatch_weighting:
                # 故意的反例：把不同样本数的 Micro-batch 等权平均。
                loss = squared_error.mean() / len(batches)
            else:
                # 正确的样本平均：所有 Micro-batch 使用同一个全局分母。
                loss = squared_error.sum() / len(inputs)
            loss.backward()
    if model.weight.grad is None:
        raise RuntimeError("Expected a non-null gradient.")
    return model.weight.grad.detach().clone()


def main() -> None:
    inputs, targets = make_data()
    full = compute_gradient(inputs, targets)
    accumulated = compute_gradient(inputs, targets, micro_batch_size=2)
    incorrect = compute_gradient(
        inputs,
        targets,
        micro_batch_size=2,
        wrong_equal_microbatch_weighting=True,
    )
    assert torch.allclose(full, accumulated, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(full, incorrect, atol=1e-12, rtol=1e-12)
    print(f"Correct accumulation error: {(full - accumulated).abs().max().item():.3e}")
    print(f"Wrong averaging error: {(full - incorrect).abs().max().item():.3e}")


if __name__ == "__main__":
    main()
