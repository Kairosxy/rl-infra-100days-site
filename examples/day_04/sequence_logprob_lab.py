"""Response-token log-probability alignment and reduction reference lab.

The exercise in extension_tasks.md asks the learner to recreate the logic,
explain the shift, and add a counterexample for incorrect reduction.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def response_token_log_probs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    valid_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select next-token logprobs and return a response-action mask.

    Inputs:
        logits: [batch_size, seq_len, vocab_size], float
        token_ids: [batch_size, seq_len], long
        prompt_lengths: [batch_size], long, includes at least one prompt token
        valid_lengths: [batch_size], long, includes response + prompt but not pad

    Outputs:
        selected_log_probs: [batch_size, seq_len - 1]
        response_action_mask: [batch_size, seq_len - 1], bool

    Output column j represents the target token at input position j + 1.
    """
    if logits.ndim != 3 or token_ids.ndim != 2:
        raise ValueError("Expected logits [B,L,V] and token_ids [B,L].")
    batch_size, seq_len, _ = logits.shape
    if token_ids.shape != (batch_size, seq_len):
        raise ValueError("logits and token_ids must have matching batch/length.")
    if prompt_lengths.shape != (batch_size,) or valid_lengths.shape != (batch_size,):
        raise ValueError("Length tensors must have shape [B].")
    if seq_len < 2 or torch.any(prompt_lengths < 1) or torch.any(
        valid_lengths < prompt_lengths
    ) or torch.any(valid_lengths > seq_len):
        raise ValueError("Invalid prompt/valid sequence lengths.")
    if torch.any(token_ids < 0) or torch.any(token_ids >= logits.shape[-1]):
        raise ValueError("Token IDs must be valid vocabulary indices.")

    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    selected = log_probs.gather(-1, token_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    target_positions = torch.arange(1, seq_len, device=token_ids.device).unsqueeze(0)
    mask = (target_positions >= prompt_lengths.unsqueeze(1)) & (
        target_positions < valid_lengths.unsqueeze(1)
    )
    return selected, mask


def reduce_response_loss(
    log_probs: torch.Tensor,
    mask: torch.Tensor,
    advantages: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Compare three mathematically distinct response loss aggregations."""
    if log_probs.ndim != 2 or mask.shape != log_probs.shape:
        raise ValueError("log_probs and mask must have matching [B,T] shapes.")
    if advantages.shape != (log_probs.shape[0],):
        raise ValueError("advantages must have shape [B].")

    masked_loss = -log_probs * advantages.detach().unsqueeze(1) * mask
    if reduction == "token_mean":
        return masked_loss.sum() / mask.sum().clamp_min(1)
    if reduction == "seq_mean_token_sum":
        return masked_loss.sum(dim=1).mean()
    if reduction == "seq_mean_token_mean":
        return (masked_loss.sum(dim=1) / mask.sum(dim=1).clamp_min(1)).mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def main() -> None:
    torch.manual_seed(42)
    logits = torch.randn(2, 7, 9, dtype=torch.float64, requires_grad=True)
    token_ids = torch.tensor(
        [[1, 3, 2, 8, 4, 0, 0], [4, 6, 1, 2, 5, 3, 0]], dtype=torch.long
    )
    selected, mask = response_token_log_probs(
        logits,
        token_ids,
        prompt_lengths=torch.tensor([2, 3]),
        valid_lengths=torch.tensor([5, 6]),
    )
    loss = reduce_response_loss(
        selected, mask, advantages=torch.tensor([1.0, -1.0]), reduction="token_mean"
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert mask.sum().item() == 6
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print(f"selected logprobs shape: {tuple(selected.shape)}")
    print(f"effective response tokens: {mask.sum().item()}")
    print(f"loss: {loss.item():.6f}")
    print("Alignment and reduction checks passed.")


if __name__ == "__main__":
    main()
