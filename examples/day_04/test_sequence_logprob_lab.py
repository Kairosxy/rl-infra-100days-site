import unittest

import torch

from sequence_logprob_lab import reduce_response_loss, response_token_log_probs


class SequenceLogprobLabTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.logits = torch.randn(2, 7, 8, dtype=torch.float64, requires_grad=True)
        self.token_ids = torch.tensor(
            [[1, 3, 2, 7, 4, 0, 0], [4, 6, 1, 2, 5, 3, 0]], dtype=torch.long
        )
        self.prompts = torch.tensor([2, 3])
        self.valid = torch.tensor([5, 6])

    def test_response_mask_matches_variable_prompt_and_padding_lengths(self):
        selected, mask = response_token_log_probs(
            self.logits, self.token_ids, self.prompts, self.valid
        )
        self.assertEqual(tuple(selected.shape), (2, 6))
        self.assertEqual(mask.tolist(), [
            [False, True, True, True, False, False],
            [False, False, True, True, True, False],
        ])

    def test_teacher_forcing_shift_selects_correct_previous_logits(self):
        selected, mask = response_token_log_probs(
            self.logits, self.token_ids, self.prompts, self.valid
        )
        reference = self.logits[:, :-1].log_softmax(-1).gather(
            -1, self.token_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        self.assertTrue(torch.allclose(selected[mask], reference[mask]))
        # 故意偏移一位：用当前位置的 Logits 预测当前位置的 Token。
        wrong = self.logits[:, 1:].log_softmax(-1).gather(
            -1, self.token_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
        self.assertFalse(torch.allclose(selected[mask], wrong[mask]))

    def test_padding_and_prompt_logits_do_not_receive_loss_gradients(self):
        selected, mask = response_token_log_probs(
            self.logits, self.token_ids, self.prompts, self.valid
        )
        loss = reduce_response_loss(selected, mask, torch.ones(2), "token_mean")
        loss.backward()
        self.assertEqual(self.logits.grad[0, 4:].abs().sum().item(), 0.0)
        self.assertEqual(self.logits.grad[1, 5:].abs().sum().item(), 0.0)
        self.assertEqual(self.logits.grad[0, :1].abs().sum().item(), 0.0)

    def test_reduction_changes_sequence_gradient_weight(self):
        mask = torch.tensor([[1, 0, 0], [1, 1, 1]], dtype=torch.bool)
        weight_ratio = {}
        for reduction in ["token_mean", "seq_mean_token_sum", "seq_mean_token_mean"]:
            selected = torch.full((2, 3), -1.0, dtype=torch.float64, requires_grad=True)
            loss = reduce_response_loss(selected, mask, torch.ones(2), reduction)
            loss.backward()
            weight_ratio[reduction] = (
                selected.grad[0].abs().sum() / selected.grad[1].abs().sum()
            ).item()
        self.assertAlmostEqual(weight_ratio["token_mean"], 1 / 3)
        self.assertAlmostEqual(weight_ratio["seq_mean_token_sum"], 1 / 3)
        self.assertAlmostEqual(weight_ratio["seq_mean_token_mean"], 1.0)

    def test_invalid_length_contract_fails(self):
        with self.assertRaisesRegex(ValueError, "Invalid"):
            response_token_log_probs(
                self.logits, self.token_ids, torch.tensor([0, 3]), self.valid
            )

    def test_advantage_is_not_differentiated(self):
        selected = torch.tensor([[-0.5, -0.2]], requires_grad=True)
        advantages = torch.tensor([1.0], requires_grad=True)
        loss = reduce_response_loss(
            selected, torch.ones_like(selected, dtype=torch.bool), advantages, "token_mean"
        )
        loss.backward()
        self.assertIsNone(advantages.grad)

    def test_all_zero_mask_has_finite_zero_loss(self):
        selected = torch.tensor([[-0.1, -0.3]], requires_grad=True)
        loss = reduce_response_loss(
            selected, torch.zeros_like(selected, dtype=torch.bool), torch.ones(1), "token_mean"
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(selected.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
