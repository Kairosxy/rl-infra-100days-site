import tempfile
import unittest
from pathlib import Path

import torch

from toy_reinforce import (
    Sample, ToyPolicy, join_rewards, load_checkpoint,
    rollout, save_checkpoint, train_step,
)


class ToyReinforceTest(unittest.TestCase):
    def setUp(self):
        self.policy = ToyPolicy()
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=0.1, weight_decay=0
        )

    def test_reward_arrival_order_does_not_change_alignment(self):
        samples = rollout(self.policy, [0, 1, 0, 1], 0, seed=7)
        rewards = [(sample.trajectory_id, float(index)) for index, sample in enumerate(samples)]
        expected = join_rewards(samples, rewards)
        shuffled = join_rewards(samples, list(reversed(rewards)))
        self.assertTrue(torch.equal(expected, shuffled))

    def test_duplicate_and_unknown_reward_ids_fail(self):
        samples = rollout(self.policy, [0, 1], 0, seed=7)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            join_rewards(samples, [("t-0", 1.0), ("t-0", 0.0)])
        with self.assertRaisesRegex(ValueError, "do not match"):
            join_rewards(samples, [("t-0", 1.0), ("unknown", 0.0)])

    def test_one_policy_step_moves_rewarded_action_probability_up(self):
        samples = [
            Sample("t-0", 0, 0, -0.693, 0),
            Sample("t-1", 0, 1, -0.693, 0),
        ]
        before = self.policy.logits.detach().clone()
        loss, next_version = train_step(
            self.policy, self.optimizer, samples,
            torch.tensor([1.0, 0.0], dtype=torch.float64), 0
        )
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertEqual(next_version, 1)
        self.assertFalse(torch.equal(before, self.policy.logits.detach()))
        self.assertGreater(self.policy.logits[0, 0].item(), self.policy.logits[0, 1].item())

    def test_stale_behavior_version_rejected(self):
        samples = rollout(self.policy, [0, 1], 0, seed=7)
        with self.assertRaisesRegex(ValueError, "Unexpected behavior"):
            train_step(self.policy, self.optimizer, samples, torch.ones(2), 1)

    def test_checkpoint_roundtrip_restores_optimizer_state(self):
        samples = rollout(self.policy, [0, 1, 0, 1], 0, seed=7)
        rewards = join_rewards(
            samples, [(item.trajectory_id, float(item.action == item.state))
                      for item in samples],
        )
        _, version = train_step(self.policy, self.optimizer, samples, rewards, 0)
        with tempfile.TemporaryDirectory() as dirname:
            path = Path(dirname) / "model.pt"
            save_checkpoint(path, self.policy, self.optimizer, version)
            restored = ToyPolicy()
            restored_optimizer = torch.optim.AdamW(
                restored.parameters(), lr=0.1, weight_decay=0
            )
            self.assertEqual(load_checkpoint(path, restored, restored_optimizer), version)
            self.assertTrue(torch.equal(self.policy.logits, restored.logits))
            self.assertEqual(
                self.optimizer.state[self.policy.logits]["step"].item(),
                restored_optimizer.state[restored.logits]["step"].item(),
            )


if __name__ == "__main__":
    unittest.main()
