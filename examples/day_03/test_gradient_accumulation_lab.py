import unittest

import torch

from gradient_accumulation_lab import compute_gradient, make_data, make_model


class GradientAccumulationTest(unittest.TestCase):
    def test_global_mean_accumulation_matches_full_batch(self):
        inputs, targets = make_data()
        expected = compute_gradient(inputs, targets)
        actual = compute_gradient(inputs, targets, micro_batch_size=2)
        self.assertTrue(torch.allclose(expected, actual, atol=1e-12, rtol=1e-12))

    def test_equal_weighting_uneven_microbatches_is_wrong(self):
        inputs, targets = make_data()
        expected = compute_gradient(inputs, targets)
        incorrect = compute_gradient(
            inputs,
            targets,
            micro_batch_size=2,
            wrong_equal_microbatch_weighting=True,
        )
        self.assertFalse(torch.allclose(expected, incorrect, atol=1e-12, rtol=1e-12))

    def test_microbatch_step_does_not_match_single_full_batch_step(self):
        inputs, targets = make_data()
        full_model = make_model()
        micro_model = make_model()
        full_optimizer = torch.optim.AdamW(full_model.parameters(), lr=0.05, weight_decay=0)
        micro_optimizer = torch.optim.AdamW(micro_model.parameters(), lr=0.05, weight_decay=0)

        full_optimizer.zero_grad()
        (full_model(inputs) - targets).square().mean().backward()
        full_optimizer.step()
        for index in range(0, len(inputs), 2):
            micro_optimizer.zero_grad()
            (micro_model(inputs[index : index + 2]) - targets[index : index + 2]).square().mean().backward()
            micro_optimizer.step()
        self.assertFalse(torch.allclose(full_model.weight, micro_model.weight, atol=1e-12, rtol=1e-12))
        self.assertEqual(full_optimizer.state[full_model.weight]["step"].item(), 1)
        self.assertEqual(micro_optimizer.state[micro_model.weight]["step"].item(), 3)


if __name__ == "__main__":
    unittest.main()
