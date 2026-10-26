import unittest

from memory_budget import estimate_adamw_state_bytes, estimate_attention_score_bytes


class MemoryBudgetTest(unittest.TestCase):
    def test_attention_quadruples_when_sequence_doubles(self):
        base = estimate_attention_score_bytes(2, 16, 640)
        doubled = estimate_attention_score_bytes(2, 16, 1280)
        self.assertEqual(doubled, base * 4)

    def test_attention_640_to_8320_scales_by_169(self):
        base = estimate_attention_score_bytes(2, 16, 640)
        large = estimate_attention_score_bytes(2, 16, 8320)
        self.assertEqual(large, base * 169)

    def test_invalid_attention_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            estimate_attention_score_bytes(2, 16, 0)

    def test_adamw_state_is_12_gb_before_activations(self):
        result = estimate_adamw_state_bytes(1_000_000_000)
        self.assertEqual(result["total"], 12_000_000_000)

    def test_master_weights_increase_budget(self):
        result = estimate_adamw_state_bytes(1_000_000_000, master_weight_bytes=4)
        self.assertEqual(result["total"], 16_000_000_000)


if __name__ == "__main__":
    unittest.main()
