"""Minimal CPU RL reference: rollout -> reward -> REINFORCE -> version.

Run from repository root: python reference_system/toy_reinforce.py
This is a contextual bandit demonstration, not PPO/GRPO training.
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Sample:
    trajectory_id: str
    state: int
    action: int
    behavior_logprob: float
    behavior_version: int


class ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.float64))


def rollout(policy: ToyPolicy, states: list[int], version: int, seed: int) -> list[Sample]:
    """Sample actions from a frozen behavior policy snapshot."""
    generator = torch.Generator().manual_seed(seed)
    samples: list[Sample] = []
    with torch.no_grad():
        for index, state in enumerate(states):
            if state not in (0, 1):
                raise ValueError("Invalid state.")
            probabilities = policy.logits[state].softmax(-1)
            action = int(torch.multinomial(probabilities, 1, generator=generator))
            samples.append(Sample(
                f"t-{index}", state, action, float(probabilities[action].log()), version
            ))
    return samples


def join_rewards(samples: list[Sample], reward_results: list[tuple[str, float]]) -> torch.Tensor:
    """Return rewards in trajectory order, independent of arrival order."""
    ids = [sample.trajectory_id for sample in samples]
    result_ids = [trajectory_id for trajectory_id, _ in reward_results]
    if len(set(ids)) != len(ids) or len(set(result_ids)) != len(result_ids):
        raise ValueError("Duplicate trajectory ID.")
    if set(ids) != set(result_ids):
        raise ValueError("Reward/trajectory IDs do not match.")
    by_id = dict(reward_results)
    rewards = torch.tensor([by_id[trajectory_id] for trajectory_id in ids], dtype=torch.float64)
    if not bool(torch.isfinite(rewards).all()):
        raise ValueError("Non-finite rewards.")
    return rewards


def train_step(
    policy: ToyPolicy,
    optimizer: torch.optim.Optimizer,
    samples: list[Sample],
    rewards: torch.Tensor,
    expected_version: int,
) -> tuple[float, int]:
    """Re-evaluate sampled actions at current policy and update once."""
    if not samples or rewards.shape != (len(samples),):
        raise ValueError("Invalid training batch.")
    if any(item.behavior_version != expected_version for item in samples):
        raise ValueError("Unexpected behavior policy version.")
    states = torch.tensor([item.state for item in samples])
    actions = torch.tensor([item.action for item in samples])
    current_logprobs = F.log_softmax(policy.logits[states], dim=-1).gather(
        1, actions.unsqueeze(1)
    ).squeeze(1)
    # 不通过环境奖励反向传播；由当前 Logprob 提供梯度路径。
    advantages = rewards.detach() - 0.5
    loss = -(current_logprobs * advantages).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(policy.logits.grad).all()):
        raise ValueError("Non-finite loss or gradient.")
    optimizer.step()
    return float(loss.detach().item()), expected_version + 1


def save_checkpoint(
    path: str | Path, policy: ToyPolicy, optimizer: torch.optim.Optimizer, version: int
) -> None:
    torch.save({
        "policy": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "version": version,
    }, path)


def load_checkpoint(
    path: str | Path, policy: ToyPolicy, optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    policy.load_state_dict(checkpoint["policy"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["version"])


def main() -> None:
    policy = ToyPolicy()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.1, weight_decay=0)
    version = 0
    for step in range(4):
        samples = rollout(policy, [0, 1] * 8, version, seed=step + 42)
        verifier_results = [
            (item.trajectory_id, float(item.action == item.state)) for item in samples
        ]
        rewards = join_rewards(samples, list(reversed(verifier_results)))
        loss, version = train_step(policy, optimizer, samples, rewards, version)
        print(f"step={step} version={version} reward={rewards.mean():.2f} loss={loss:.4f}")
    with tempfile.TemporaryDirectory() as folder:
        save_checkpoint(Path(folder) / "snapshot.pt", policy, optimizer, version)
        restored_policy = ToyPolicy()
        restored_optimizer = torch.optim.AdamW(
            restored_policy.parameters(), lr=0.1, weight_decay=0
        )
        restored_version = load_checkpoint(
            Path(folder) / "snapshot.pt", restored_policy, restored_optimizer
        )
        assert restored_version == version
        assert torch.equal(restored_policy.logits, policy.logits)
        print(f"Checkpoint restored at policy version {restored_version}.")


if __name__ == "__main__":
    main()
