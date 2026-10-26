import unittest

from trajectory_contract import (
    RewardResult,
    Trajectory,
    join_rewards_by_trajectory_id,
    validate_trajectories,
)


def make_group(group_id: str, count: int = 4, policy_version: int = 10) -> list[Trajectory]:
    return [
        Trajectory(
            trajectory_id=f"{group_id}-{index}",
            group_id=group_id,
            response_ids=[1, 2, 3],
            response_mask=[1, 1, 1],
            reward=0.0,
            policy_version=policy_version,
        )
        for index in range(count)
    ]


class TrajectoryContractTest(unittest.TestCase):
    def test_valid_batch_passes(self):
        validate_trajectories(make_group("g0") + make_group("g1"))

    def test_incomplete_group_fails(self):
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            validate_trajectories(make_group("g0", count=3))

    def test_duplicate_trajectory_id_fails(self):
        trajectories = make_group("g0")
        trajectories[1] = trajectories[0]
        with self.assertRaisesRegex(ValueError, "Duplicate trajectory_id"):
            validate_trajectories(trajectories)

    def test_mask_length_mismatch_fails(self):
        trajectories = make_group("g0")
        trajectories[0] = Trajectory(
            trajectory_id="g0-0",
            group_id="g0",
            response_ids=[1, 2],
            response_mask=[1],
            reward=1.0,
            policy_version=10,
        )
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            validate_trajectories(trajectories)

    def test_nan_reward_fails(self):
        trajectories = make_group("g0")
        trajectories[0] = Trajectory(
            trajectory_id="g0-0",
            group_id="g0",
            response_ids=[1, 2, 3],
            response_mask=[1, 1, 1],
            reward=float("nan"),
            policy_version=10,
        )
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_trajectories(trajectories)

    def test_mixed_policy_versions_in_group_fail_by_default(self):
        trajectories = make_group("g0")
        trajectories[3] = Trajectory(
            trajectory_id="g0-3",
            group_id="g0",
            response_ids=[1, 2, 3],
            response_mask=[1, 1, 1],
            reward=0.0,
            policy_version=11,
        )
        with self.assertRaisesRegex(ValueError, "Mixed policy versions"):
            validate_trajectories(trajectories)

    def test_mixed_versions_can_be_explicitly_allowed_for_async_experiments(self):
        trajectories = make_group("g0")
        trajectories[3] = Trajectory(
            trajectory_id="g0-3",
            group_id="g0",
            response_ids=[1, 2, 3],
            response_mask=[1, 1, 1],
            reward=0.0,
            policy_version=11,
        )
        validate_trajectories(
            trajectories,
            require_group_version_consistency=False,
        )

    def test_reward_join_is_independent_of_result_order(self):
        trajectories = make_group("g0") + make_group("g1")
        rewards = [
            RewardResult(item.trajectory_id, float(index))
            for index, item in enumerate(trajectories)
        ]
        joined_forward = join_rewards_by_trajectory_id(trajectories, rewards)
        joined_reverse = join_rewards_by_trajectory_id(trajectories, list(reversed(rewards)))
        self.assertEqual(
            [item.reward for item in joined_forward],
            [item.reward for item in joined_reverse],
        )

    def test_reward_join_rejects_missing_reward(self):
        trajectories = make_group("g0")
        rewards = [RewardResult(item.trajectory_id, 1.0) for item in trajectories[:-1]]
        with self.assertRaisesRegex(ValueError, "Missing rewards"):
            join_rewards_by_trajectory_id(trajectories, rewards)

    def test_reward_join_rejects_unknown_id(self):
        trajectories = make_group("g0")
        rewards = [RewardResult(item.trajectory_id, 1.0) for item in trajectories]
        rewards.append(RewardResult("unknown-id", 1.0))
        with self.assertRaisesRegex(ValueError, "Unknown reward"):
            join_rewards_by_trajectory_id(trajectories, rewards)

    def test_reward_join_rejects_duplicate_reward_id(self):
        trajectories = make_group("g0")
        rewards = [RewardResult(item.trajectory_id, 1.0) for item in trajectories]
        rewards.append(rewards[0])
        with self.assertRaisesRegex(ValueError, "Duplicate reward"):
            join_rewards_by_trajectory_id(trajectories, rewards)


if __name__ == "__main__":
    unittest.main()
