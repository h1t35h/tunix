# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from tunix.rl import advantage_shaping
from tunix.rl import algorithm_config
from tunix.rl import function_registry


class AdvantageShapingTest(parameterized.TestCase):

  def test_length_penalty_below_or_equal_threshold_noop(self):
    rewards = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    pass_mask = rewards > 0
    lengths = np.array([100, 500, 200, 300], dtype=np.int32)
    penalized, stats = advantage_shaping.group_relative_length_penalty(
        rewards,
        pass_mask,
        lengths,
        num_generations=4,
        lp_pass_rate_threshold=0.5,
    )
    np.testing.assert_allclose(penalized, rewards)
    self.assertEqual(stats["length_penalty/applied_frac"], 0.0)
    self.assertEqual(stats["length_penalty/mean"], 0.0)

  def test_length_penalty_above_threshold_curve(self):
    # 3 of 4 pass -> pass_rate = 0.75 > 0.5. Shortest passing is l* = 100.
    # Trajectory 0: l=100 -> ratio=1.0 (in dead zone <= 1.2) -> penalty 0.0
    # Trajectory 1: l=160 -> ratio=1.6 -> (1.6 - 1.2)/(1.0 - 0.2) = 0.5 -> penalty 0.2
    # Trajectory 2: l=300 -> ratio=3.0 -> clipped to 1.0 -> penalty 0.4
    # Trajectory 3: fail -> penalty 0.0
    rewards = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    pass_mask = rewards > 0
    lengths = np.array([100, 160, 300, 50], dtype=np.int32)
    penalized, stats = advantage_shaping.group_relative_length_penalty(
        rewards,
        pass_mask,
        lengths,
        num_generations=4,
        lp_pass_rate_threshold=0.5,
        lp_max_penalty=0.4,
        lp_dead_zone=0.2,
        lp_saturation=1.0,
        lp_gamma=1.0,
    )
    np.testing.assert_allclose(
        penalized,
        np.array([1.0, 0.8, 0.6, 0.0], dtype=np.float32),
        atol=1e-6,
    )
    self.assertAlmostEqual(stats["length_penalty/applied_frac"], 1.0)
    self.assertAlmostEqual(stats["length_penalty/mean"], 0.6 / 3.0, places=6)

  def test_quality_from_ranks_distinct_and_ties(self):
    pass_mask = np.array([True, True, True, False])
    ranks = np.array([1, 2, 3, 4], dtype=np.int32)
    q = advantage_shaping.quality_from_ranks(
        ranks, pass_mask, num_generations=4, rho=0.5
    )
    np.testing.assert_allclose(
        q, np.array([1.0, 0.75, 0.5, 1.0], dtype=np.float32), atol=1e-6
    )

    # Ties: first two tie -> mean rank 1.5 -> f = 1 - 0.5 * (0.5 / 2) = 0.875
    ranks_tied = np.array([1, 1, 3, 4], dtype=np.int32)
    q_tied = advantage_shaping.quality_from_ranks(
        ranks_tied, pass_mask, num_generations=4, rho=0.5
    )
    np.testing.assert_allclose(
        q_tied,
        np.array([0.875, 0.875, 0.5, 1.0], dtype=np.float32),
        atol=1e-6,
    )

  def test_gar_zero_sum_and_lambda_cap_across_100_random_groups(self):
    """Gate G3 Invariant #2: GAR zero-sum (< 1e-5) & lambda <= lambda_max across 100 random groups."""
    rng = np.random.default_rng(42)
    num_generations = 8
    lambda_max = 2.0

    for group_idx in range(100):
      # Cover |P| in {0, 1, G} as well as general subsets and length-penalized A_i <= 0
      if group_idx % 10 == 0:
        pass_mask = np.zeros(num_generations, dtype=bool)
      elif group_idx % 10 == 1:
        pass_mask = np.zeros(num_generations, dtype=bool)
        pass_mask[0] = True
      elif group_idx % 10 == 2:
        pass_mask = np.ones(num_generations, dtype=bool)
      else:
        pass_mask = rng.random(num_generations) > 0.35

      # Generate rewards with optional severe length penalties so some i in P have A_i <= 0
      rewards = np.where(pass_mask, 1.0, 0.0).astype(np.float32)
      if group_idx % 5 == 0 and np.any(pass_mask):
        # Penalize one passing trajectory below group mean so its A_i <= 0
        first_pass = int(np.flatnonzero(pass_mask)[0])
        rewards[first_pass] = 0.1

      mean_r = float(np.mean(rewards))
      std_r = float(np.std(rewards, ddof=1))
      if std_r > 1e-6:
        adv = ((rewards - mean_r) / (std_r + 1e-6)).astype(np.float32)
      else:
        adv = np.zeros(num_generations, dtype=np.float32)

      ranks = rng.integers(1, num_generations + 1, size=num_generations)
      qualities = advantage_shaping.quality_from_ranks(
          ranks, pass_mask, num_generations=num_generations, rho=0.5
      )
      new_adv, stats = advantage_shaping.groupwise_advantage_redistribution(
          adv,
          pass_mask,
          qualities,
          num_generations=num_generations,
          gar_lambda_max=lambda_max,
      )
      self.assertLess(
          abs(float(np.sum(new_adv))),
          1e-5,
          msg=f"Group {group_idx} failed zero-sum invariant: sum={np.sum(new_adv)}",
      )
      self.assertLessEqual(
          stats["advantage/gar_lambda_mean"],
          lambda_max + 1e-6,
          msg=f"Group {group_idx} exceeded lambda_max: {stats}",
      )

  def test_span_level_token_advantages_conservation_and_bounds(self):
    # Group of G=2: trajectory 0 has A = +1.0, trajectory 1 has A = -1.0
    # T = 4 active tokens each: 1 redundant token (h=1) and 3 clean tokens (h=0)
    seq_adv = np.array([1.0, -1.0], dtype=np.float32)
    comp_mask = np.array(
        [[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=np.float32
    )
    red_mask = np.array([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.int8)

    # Default span_beta_pos=0.5:
    # Positive trajectory: 1 redundant, 3 clean -> alpha = 1 + 0.5*(1/3) = 7/6
    tok_adv, stats = advantage_shaping.span_level_token_advantages(
        seq_adv,
        red_mask,
        comp_mask,
        num_generations=2,
        span_kappa=1.5,
        span_alpha_max=2.0,
        span_beta_min=0.5,
    )
    self.assertAlmostEqual(stats["advantage/span_alpha_mean"], 7.0 / 6.0, places=5)
    self.assertAlmostEqual(stats["advantage/span_beta_mean"], 5.0 / 6.0, places=5)
    self.assertAlmostEqual(float(tok_adv[0, 1]), 0.5, places=5)
    self.assertAlmostEqual(float(tok_adv[0, 0]), 7.0 / 6.0, places=5)
    self.assertEqual(float(tok_adv[0, 4]), 0.0)
    self.assertEqual(float(tok_adv[1, 4]), 0.0)
    self.assertAlmostEqual(float(tok_adv[1, 2]), -1.5, places=5)
    self.assertAlmostEqual(float(np.sum(tok_adv[0, :4])), 4.0 * 1.0, places=5)
    self.assertAlmostEqual(float(np.sum(tok_adv[1, :4])), 4.0 * (-1.0), places=5)

    # Explicit span_beta_pos=0.0: redundant token on positive trajectory is 0.0, alpha = 4/3
    tok_adv_zero, stats_zero = advantage_shaping.span_level_token_advantages(
        seq_adv,
        red_mask,
        comp_mask,
        num_generations=2,
        span_kappa=1.5,
        span_alpha_max=2.0,
        span_beta_min=0.5,
        span_beta_pos=0.0,
    )
    self.assertAlmostEqual(
        stats_zero["advantage/span_alpha_mean"], 4.0 / 3.0, places=5
    )
    self.assertEqual(float(tok_adv_zero[0, 1]), 0.0)

  def test_span_level_token_advantages_mass_conservation_when_clipped(self):
    # High flagging rate (5 of 6 tokens flagged) causes alpha to hit span_alpha_max
    # and beta to hit span_beta_min. Verify span_renormalize=True restores exact
    # group positive and negative token advantage mass.
    seq_adv = np.array([1.2, -1.2], dtype=np.float32)
    comp_mask = np.ones((2, 6), dtype=np.float32)
    red_mask = np.array(
        [[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]], dtype=np.int8
    )
    tok_adv_renorm, stats_renorm = (
        advantage_shaping.span_level_token_advantages(
            seq_adv,
            red_mask,
            comp_mask,
            num_generations=2,
            span_kappa=1.5,
            span_alpha_max=1.5,
            span_beta_min=0.5,
            span_beta_pos=0.5,
            span_renormalize=True,
        )
    )
    self.assertAlmostEqual(
        float(np.sum(tok_adv_renorm[0])), 6.0 * 1.2, places=4
    )
    self.assertAlmostEqual(
        float(np.sum(tok_adv_renorm[1])), 6.0 * (-1.2), places=4
    )
    self.assertAlmostEqual(
        stats_renorm["advantage/mass_pos_ratio"], 1.0, places=4
    )
    self.assertAlmostEqual(
        stats_renorm["advantage/mass_neg_ratio"], 1.0, places=4
    )

  def test_hard_mask_token_advantages(self):
    seq_adv = np.array([1.5, -0.5], dtype=np.float32)
    comp_mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.float32)
    red_mask = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int8)
    out = advantage_shaping.hard_mask_token_advantages(
        seq_adv, red_mask, comp_mask
    )
    expected = np.array(
        [[1.5, 0.0, 0.0], [0.0, -0.5, -0.5]], dtype=np.float32
    )
    np.testing.assert_allclose(out, expected, atol=1e-6)

  def test_registered_mimo_estimator_end_to_end(self):
    estimator = function_registry.get_advantage_estimator("mimo")
    rewards = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    comp_mask = np.ones((4, 6), dtype=np.float32)
    red_mask = np.zeros((4, 6), dtype=np.int8)
    red_mask[0, 1] = 1
    red_mask[3, 2] = 1
    ranks = np.array([1, 2, 3, 4], dtype=np.int32)
    lengths = np.array([100, 150, 220, 180], dtype=np.int32)
    stats_out: dict[str, Any] = {}

    tok_adv = estimator(
        rewards,
        num_generations=4,
        completion_mask=comp_mask,
        token_flags=red_mask,
        passed=rewards > 0,
        ranks=ranks,
        lengths=lengths,
        cfg=algorithm_config.MimoAdvantageConfig(),
        stats_out=stats_out,
    )
    self.assertEqual(tok_adv.shape, (4, 6))
    self.assertEqual(tok_adv.dtype, np.float32)
    self.assertGreater(float(tok_adv[0, 1]), 0.0)
    self.assertLess(float(tok_adv[0, 1]), float(tok_adv[0, 0]))
    self.assertIn("length_penalty/mean", stats_out)
    self.assertIn("advantage/gar_lambda_mean", stats_out)
    self.assertIn("advantage/span_alpha_mean", stats_out)
    self.assertIn("advantage/span_beta_mean", stats_out)
    self.assertIn("advantage/mass_pos_ratio", stats_out)
    self.assertIn("advantage/mass_neg_ratio", stats_out)


if __name__ == "__main__":
  absltest.main()
