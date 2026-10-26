import unittest

import torch

from policy_gradient_lab import (
    policy_gradient_loss,
    selected_action_log_probs,
)


class PolicyGradientLabTest(unittest.TestCase):
    def test_selected_log_probs_match_log_softmax(self):
        logits = torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float64)
        actions = torch.tensor([1])
        expected = logits.log_softmax(dim=-1)[0, 1]
        actual = selected_action_log_probs(logits, actions)[0]
        self.assertTrue(torch.allclose(actual, expected))

    def test_masked_sample_has_zero_gradient(self):
        logits = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        actions = torch.tensor([0, 1])
        advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
        loss = policy_gradient_loss(
            logits,
            actions,
            advantages,
            mask=torch.tensor([1.0, 0.0], dtype=torch.float64),
        )
        loss.backward()
        self.assertTrue(torch.allclose(logits.grad[1], torch.zeros_like(logits.grad[1])))

    def test_symmetric_advantages_can_cancel_loss_but_not_gradient(self):
        logits = torch.tensor(
            [[2.0, 0.0], [0.0, 2.0]],
            dtype=torch.float64,
            requires_grad=True,
        )
        actions = torch.tensor([0, 1])
        advantages = torch.tensor([1.0, -1.0], dtype=torch.float64)
        loss = policy_gradient_loss(logits, actions, advantages)
        loss.backward()
        self.assertAlmostEqual(loss.item(), 0.0, places=12)
        self.assertGreater(logits.grad.abs().sum().item(), 0.0)

    def test_empty_mask_returns_finite_zero_loss_and_zero_gradient(self):
        logits = torch.tensor([[1.0, 0.0]], dtype=torch.float64, requires_grad=True)
        actions = torch.tensor([0])
        advantages = torch.tensor([1.0], dtype=torch.float64)
        mask = torch.tensor([0.0], dtype=torch.float64)
        loss = policy_gradient_loss(logits, actions, advantages, mask)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 0.0, places=12)
        self.assertTrue(torch.allclose(logits.grad, torch.zeros_like(logits.grad)))

    def test_advantage_is_detached(self):
        logits = torch.tensor([[0.3, -0.1]], dtype=torch.float64, requires_grad=True)
        actions = torch.tensor([0])
        advantages = torch.tensor([1.0], dtype=torch.float64, requires_grad=True)
        loss = policy_gradient_loss(logits, actions, advantages)
        loss.backward()
        self.assertIsNone(advantages.grad)


if __name__ == "__main__":
    unittest.main()
