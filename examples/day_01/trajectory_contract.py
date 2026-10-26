"""Validate and join a small RL trajectory batch contract.

Run from the repository root:
    python examples/day_01/trajectory_contract.py
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Sequence


@dataclass(frozen=True)
class Trajectory:
    trajectory_id: str
    group_id: str
    response_ids: list[int]
    response_mask: list[int]
    reward: float
    policy_version: int


@dataclass(frozen=True)
class RewardResult:
    trajectory_id: str
    reward: float


def join_rewards_by_trajectory_id(
    trajectories: Sequence[Trajectory],
    reward_results: Sequence[RewardResult],
) -> list[Trajectory]:
    """Attach rewards by stable ID, never by list position."""
    reward_by_id: dict[str, float] = {}
    for result in reward_results:
        if result.trajectory_id in reward_by_id:
            raise ValueError(f"Duplicate reward trajectory_id: {result.trajectory_id}")
        if not math.isfinite(result.reward):
            raise ValueError(f"Reward must be finite for {result.trajectory_id}.")
        reward_by_id[result.trajectory_id] = result.reward

    trajectory_ids = {item.trajectory_id for item in trajectories}
    reward_ids = set(reward_by_id)
    missing = trajectory_ids - reward_ids
    unknown = reward_ids - trajectory_ids
    if missing:
        raise ValueError(f"Missing rewards for trajectory IDs: {sorted(missing)}")
    if unknown:
        raise ValueError(f"Unknown reward trajectory IDs: {sorted(unknown)}")

    return [replace(item, reward=reward_by_id[item.trajectory_id]) for item in trajectories]


def validate_trajectories(
    trajectories: Sequence[Trajectory],
    expected_group_size: int = 4,
    require_group_version_consistency: bool = True,
) -> None:
    """Raise ValueError when sample identity or group contracts are broken."""
    if not trajectories:
        raise ValueError("Trajectory batch must not be empty.")

    trajectory_ids = [item.trajectory_id for item in trajectories]
    if len(trajectory_ids) != len(set(trajectory_ids)):
        raise ValueError("Duplicate trajectory_id detected.")

    group_counts = Counter(item.group_id for item in trajectories)
    incomplete_groups = {
        group_id: count
        for group_id, count in group_counts.items()
        if count != expected_group_size
    }
    if incomplete_groups:
        raise ValueError(f"Incomplete or oversized groups: {incomplete_groups}")

    group_versions: dict[str, set[int]] = {}
    for item in trajectories:
        if not item.trajectory_id or not item.group_id:
            raise ValueError("trajectory_id and group_id must be non-empty.")
        if len(item.response_ids) != len(item.response_mask):
            raise ValueError("response_ids and response_mask length mismatch.")
        if any(mask not in (0, 1) for mask in item.response_mask):
            raise ValueError("response_mask values must be 0 or 1.")
        if not math.isfinite(item.reward):
            raise ValueError("Reward must be finite.")
        if item.policy_version < 0:
            raise ValueError("policy_version must be non-negative.")
        group_versions.setdefault(item.group_id, set()).add(item.policy_version)

    if require_group_version_consistency:
        mixed_versions = {
            group_id: sorted(versions)
            for group_id, versions in group_versions.items()
            if len(versions) > 1
        }
        if mixed_versions:
            raise ValueError(f"Mixed policy versions within groups: {mixed_versions}")


def main() -> None:
    trajectories = [
        Trajectory(
            trajectory_id=f"traj-{group_id}-{sample_id}",
            group_id=f"group-{group_id}",
            response_ids=[11, 12, 13],
            response_mask=[1, 1, 1],
            reward=0.0,
            policy_version=10,
        )
        for group_id in range(2)
        for sample_id in range(4)
    ]
    rewards = [
        RewardResult(trajectory_id=item.trajectory_id, reward=float(index))
        for index, item in enumerate(reversed(trajectories))
    ]
    joined = join_rewards_by_trajectory_id(trajectories, rewards)
    validate_trajectories(joined, expected_group_size=4)
    print(f"Validated {len(joined)} trajectories across 2 groups.")
    print("Reward join is ID-based; group size and policy versions are consistent.")


if __name__ == "__main__":
    main()
