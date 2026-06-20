import unittest

import numpy as np
import torch


class FakeTokenizer:
    mask_token_id = 7
    all_special_ids = [0, 7, 8]

    def __len__(self):
        return 9

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [1, 2] if text else []

    def build_inputs_with_special_tokens(self, token_ids):
        return [8] + list(token_ids) + [0]

    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        pieces = {0: "<pad>", 1: " A", 2: " B", 3: ".", 4: " C", 5: " —", 6: " ", 7: "<mask>", 8: "<s>"}
        return "".join(pieces[int(token_id)] for token_id in token_ids)


class HfDiffusionHelpersTest(unittest.TestCase):
    def test_high_mask_distribution_biases_toward_full_continuations(self) -> None:
        from scripts.train_hf_masked_diffusion import sample_mask_rates

        rng = np.random.default_rng(7)
        uniform = sample_mask_rates(
            batch_size=64,
            rng=rng,
            min_mask_ratio=0.05,
            max_mask_ratio=1.0,
            mask_distribution="uniform",
            full_mask_fraction=0.0,
            high_mask_power=2.0,
        )
        rng = np.random.default_rng(7)
        high = sample_mask_rates(
            batch_size=64,
            rng=rng,
            min_mask_ratio=0.05,
            max_mask_ratio=1.0,
            mask_distribution="high",
            full_mask_fraction=0.25,
            high_mask_power=3.0,
        )
        self.assertGreater(float(high.mean()), float(uniform.mean()))
        self.assertGreaterEqual(int(np.count_nonzero(high == 1.0)), 1)

    def test_entropy_bound_commits_at_least_scheduled_progress(self) -> None:
        from scripts.sample_hf_masked_diffusion import entropy_bound_commit_count

        entropy = np.array([0.01, 0.02, 0.6, 0.7], dtype=np.float64)
        self.assertEqual(entropy_bound_commit_count(entropy, entropy_bound=0.05, min_commit=1), 2)
        self.assertEqual(entropy_bound_commit_count(entropy, entropy_bound=0.0, min_commit=3), 3)

    def test_repetition_penalty_marks_over_repeated_pieces(self) -> None:
        from scripts.sample_hf_masked_diffusion import repetition_penalty_scores

        penalties = repetition_penalty_scores(
            tokenizer=FakeTokenizer(),
            sampled=np.array([1, 1, 1, 2, 3, 4]),
            repeat_penalty=2.0,
            max_fraction=0.25,
        )
        self.assertGreater(float(penalties[0]), 0.0)
        self.assertEqual(float(penalties[3]), 0.0)

    def test_piece_logit_penalty_marks_blank_and_bad_punctuation(self) -> None:
        from scripts.sample_hf_masked_diffusion import piece_logit_penalties

        penalties = piece_logit_penalties(FakeTokenizer(), vocab_size=7, penalty=1.5)
        assert penalties is not None
        self.assertEqual(float(penalties[1]), 0.0)
        self.assertEqual(float(penalties[3]), 0.0)
        self.assertGreater(float(penalties[5]), 0.0)
        self.assertGreater(float(penalties[6]), 0.0)

    def test_quality_score_prefers_diverse_continuations(self) -> None:
        from scripts.sample_hf_masked_diffusion import quality_score

        prompt = "Prompt."
        good = prompt + " The afternoon light moved across the office while the voices settled into another room."
        bad = prompt + " and all of the —. and all of the —. and all of the —."
        self.assertGreater(quality_score(good, prompt), quality_score(bad, prompt))

    def test_uniform_corruption_replaces_only_selected_positions(self) -> None:
        from scripts.train_hf_masked_diffusion import corrupt_core_tokens

        clean = np.array([[1, 2, 3, 4]], dtype=np.int64)
        mask = np.array([[False, True, False, True]])
        noisy = corrupt_core_tokens(
            clean_core=clean,
            mask_core=mask,
            tokenizer=FakeTokenizer(),
            rng=np.random.default_rng(3),
            corruption="uniform",
            uniform_corruption_fraction=1.0,
        )
        self.assertEqual(int(noisy[0, 0]), 1)
        self.assertEqual(int(noisy[0, 2]), 3)
        self.assertNotEqual(int(noisy[0, 1]), 7)
        self.assertNotEqual(int(noisy[0, 3]), 7)

    def test_uniform_noise_build_input_tracks_generated_span(self) -> None:
        from scripts.sample_hf_masked_diffusion import build_input, valid_uniform_noise_ids

        tokenizer = FakeTokenizer()
        noise_ids = valid_uniform_noise_ids(tokenizer, len(tokenizer))
        input_ids, positions = build_input(
            tokenizer=tokenizer,
            prompt="Prompt",
            new_tokens=3,
            initial_noise="uniform",
            rng=np.random.default_rng(4),
            noise_token_ids=noise_ids,
        )
        self.assertEqual(input_ids[0], 8)
        self.assertEqual(input_ids[-1], 0)
        self.assertEqual(positions.tolist(), [3, 4, 5])
        self.assertTrue(all(input_ids[index] not in tokenizer.all_special_ids for index in positions))

    def test_mdlm_loss_weighting_balances_rows_then_weights_noise_rates(self) -> None:
        from scripts.train_hf_masked_diffusion import label_loss_weights

        labels = torch.tensor([[1, -100, 2, -100], [3, 4, 5, -100]])
        rates = torch.tensor([0.25, 1.0])
        sequence = label_loss_weights(labels, rates, loss_weighting="sequence", max_loss_weight=8.0)
        mdlm = label_loss_weights(labels, rates, loss_weighting="mdlm", max_loss_weight=8.0)

        self.assertAlmostEqual(float(sequence[0].sum()), 1.0)
        self.assertAlmostEqual(float(sequence[1].sum()), 1.0)
        self.assertAlmostEqual(float(mdlm[0].sum()), 4.0)
        self.assertAlmostEqual(float(mdlm[1].sum()), 1.0)

    def test_subs_parameterization_filters_special_predictions(self) -> None:
        from scripts.train_hf_masked_diffusion import filter_for_subs_parameterization, forbidden_prediction_ids

        tokenizer = FakeTokenizer()
        logits = torch.zeros((1, 2, len(tokenizer)))
        forbidden = forbidden_prediction_ids(tokenizer, len(tokenizer))
        filtered = filter_for_subs_parameterization(logits, forbidden, objective="mdlm-subs")

        self.assertLess(float(filtered[0, 0, tokenizer.mask_token_id]), -1e8)
        self.assertLess(float(filtered[0, 0, 0]), -1e8)
        self.assertEqual(float(filtered[0, 0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
