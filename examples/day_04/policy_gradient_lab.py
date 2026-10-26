"""A small CPU-only lab for REINFORCE log-probability gradients.

Run:
    python examples/day_04/policy_gradient_lab.py

This script checks gradient direction, finite values, and masking. It is
a teaching example, not a complete RL training implementation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def selected_action_log_probs(
    logits: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Return log-probabilities for selected actions.

    Args:
        logits: Float tensor shaped [batch_size, vocab_size].
        actions: Long tensor shaped [batch_size].
    """
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=actions.unsqueeze(-1)).squeeze(-1)


def policy_gradient_loss(
    logits: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Token/sample-mean teaching loss with optional binary mask."""
    selected_log_probs = selected_action_log_probs(logits, actions)
    token_loss = -selected_log_probs * advantages.detach()

    if mask is None:
        return token_loss.mean()

    mask = mask.to(dtype=token_loss.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return (token_loss * mask).sum() / denominator


def main() -> None:
    torch.manual_seed(7)

    logits = torch.tensor(
        [[2.0, 0.0], [0.0, 2.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    actions = torch.tensor([0, 1], dtype=torch.long)
    advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)

    loss = policy_gradient_loss(logits, actions, advantages)
    # The scalar loss cancels to zero because log-probs are equal and advantages
    # are +1/-1. Its gradient is still non-zero: loss value alone is not a
    # reliable signal that the objective has no learning signal.
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    # Gradient descent increases the positive-advantage action's logit
    # and decreases the negative-advantage action's logit.
    assert logits.grad[0, 0] < 0
    assert logits.grad[1, 1] > 0

    masked_logits = logits.detach().clone().requires_grad_(True)
    masked_loss = policy_gradient_loss(
        masked_logits,
        actions,
        advantages,
        mask=torch.tensor([1.0, 0.0], dtype=torch.float64),
    )
    masked_loss.backward()
    assert masked_logits.grad is not None
    assert torch.allclose(
        masked_logits.grad[1],
        torch.zeros_like(masked_logits.grad[1]),
    )

    # Verify the analytic gradient of log-softmax for one sample:
    # d log p(a) / d z_k = 1[k == a] - p(k).
    one_logits = torch.tensor(
        [0.4, -0.2, 0.1],
        dtype=torch.float64,
        requires_grad=True,
    )
    action = torch.tensor([2])
    selected_log_prob = selected_action_log_probs(
        one_logits.unsqueeze(0),
        action,
    ).sum()
    analytic = torch.autograd.grad(selected_log_prob, one_logits)[0]
    probabilities = one_logits.detach().softmax(dim=-1)
    expected = -probabilities
    expected[2] += 1.0
    assert torch.allclose(analytic, expected, atol=1e-12, rtol=1e-12)

    print(f"loss={loss.item():.6f}")
    print(f"masked_loss={masked_loss.item():.6f}")
    print(f"max_grad_abs={logits.grad.abs().max().item():.6f}")
    print("All policy-gradient checks passed.")


if __name__ == "__main__":
    main()
