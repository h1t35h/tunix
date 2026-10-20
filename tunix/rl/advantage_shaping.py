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

"""MiMo-V2-Pro advantage shaping: Length Penalty (LP), GAR, and SPAN."""

from collections.abc import Sequence
from typing import Any
import numpy as np
from tunix.rl import algorithm_config
from tunix.rl import function_registry


def _validate_group_size(batch_size: int, num_generations: int) -> int:
  if num_generations <= 0:
    raise ValueError(
        f"num_generations must be positive, got {num_generations}."
    )
  if batch_size % num_generations != 0:
    raise ValueError(
        f"Batch size {batch_size} must be divisible by num_generations"
        f" {num_generations}."
    )
  return batch_size // num_generations


def _average_tie_ranks_1d(values: np.ndarray) -> np.ndarray:
  """Computes 1-indexed ranks with average tie-breaking (like rankdata)."""
  n = values.shape[0]
  if n == 0:
    return np.empty(0, dtype=np.float32)
  if n == 1:
    return np.ones(1, dtype=np.float32)
  order = np.argsort(values, kind="mergesort")
  sorted_vals = values[order]
  ranks = np.empty(n, dtype=np.float32)
  i = 0
  while i < n:
    j = i + 1
    while j < n and sorted_vals[j] == sorted_vals[i]:
      j += 1
    mean_rank = 0.5 * float((i + 1) + j)
    ranks[order[i:j]] = mean_rank
    i = j
  return ranks


def group_relative_length_penalty(
    rewards: np.ndarray,
    passed: np.ndarray,
    lengths: np.ndarray,
    num_generations: int,
    *,
    lp_pass_rate_threshold: float = 0.5,
    lp_max_penalty: float = 0.4,
    lp_dead_zone: float = 0.2,
    lp_saturation: float = 1.0,
    lp_gamma: float = 1.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
  """Applies MiMo-V2-Pro group-relative length penalty (Eq. LP).

  Args:
    rewards: Sequence rewards of shape `[B]`.
    passed: Boolean mask of passing trajectories (`R_i > 0`), shape `[B]`.
    lengths: Total generated response tokens per trajectory, shape `[B]`.
    num_generations: Group size `G`.
    lp_pass_rate_threshold: Threshold `A_thr` on `|P| / G` above which LP
      applies.
    lp_max_penalty: Maximum reward penalty `X`.
    lp_dead_zone: Relative length dead-zone `delta`.
    lp_saturation: Relative length saturation `s`.
    lp_gamma: Exponent `gamma` applied to clipped penalty ratio.
    valid_mask: Optional boolean mask of valid (non-padded) trajectories `[B]`.

  Returns:
    Tuple `(penalized_rewards, stats)` where `penalized_rewards` has shape `[B]`
    and `dtype=float32`.
  """
  if lp_saturation <= lp_dead_zone:
    raise ValueError(
        f"lp_saturation ({lp_saturation}) must be greater than lp_dead_zone"
        f" ({lp_dead_zone})."
    )
  rewards_f32 = np.asarray(rewards, dtype=np.float32).reshape(-1)
  passed_bool = np.asarray(passed, dtype=bool).reshape(-1)
  lengths_f32 = np.asarray(lengths, dtype=np.float32).reshape(-1)
  batch_size = rewards_f32.shape[0]
  if batch_size == 0:
    empty_stats = {
        "length_penalty/mean": 0.0,
        "length_penalty/applied_frac": 0.0,
        "mean": 0.0,
        "applied_frac": 0.0,
    }
    return rewards_f32.copy(), empty_stats

  num_groups = _validate_group_size(batch_size, num_generations)
  if valid_mask is None:
    valid_bool = np.ones(batch_size, dtype=bool)
  else:
    valid_bool = np.asarray(valid_mask, dtype=bool).reshape(-1)

  penalized = rewards_f32.copy()
  penalties = np.zeros(batch_size, dtype=np.float32)
  applied_groups = 0
  denom = float(lp_saturation - lp_dead_zone)

  for g in range(num_groups):
    start = g * num_generations
    end = start + num_generations
    g_valid = valid_bool[start:end]
    g_passed = passed_bool[start:end] & g_valid
    pass_count = int(np.sum(g_passed))
    pass_rate = float(pass_count) / float(num_generations)
    if pass_count > 0 and pass_rate > lp_pass_rate_threshold:
      g_lengths = lengths_f32[start:end]
      l_star = float(np.min(g_lengths[g_passed]))
      if l_star <= 0.0:
        continue
      applied_groups += 1
      rel_excess = (g_lengths / l_star - 1.0 - lp_dead_zone) / denom
      clipped = np.clip(rel_excess, 0.0, 1.0)
      g_pen = np.where(
          g_passed,
          float(lp_max_penalty) * np.power(clipped, float(lp_gamma)),
          0.0,
      ).astype(np.float32)
      penalties[start:end] = g_pen
      penalized[start:end] = rewards_f32[start:end] - g_pen

  passing_mask_all = passed_bool & valid_bool
  mean_penalty = (
      float(np.mean(penalties[passing_mask_all]))
      if np.any(passing_mask_all)
      else 0.0
  )
  applied_frac = float(applied_groups) / float(max(num_groups, 1))
  stats = {
      "length_penalty/mean": mean_penalty,
      "length_penalty/applied_frac": applied_frac,
      "mean": mean_penalty,
      "applied_frac": applied_frac,
  }
  return penalized, stats


def quality_from_ranks(
    ranks: Sequence[float] | np.ndarray,
    passed_mask: Sequence[bool] | np.ndarray | None = None,
    num_generations: int | None = None,
    *,
    rho: float = 0.5,
) -> np.ndarray:
  """Converts group ranks among passing trajectories into quality multipliers.

  Implements:
    `f_i = 1 - rho * (k_i - 1) / (|P| - 1)` for `i in P` with `|P| >= 2`,
  where `k_i in [1, |P|]` is the 1-indexed within-`P` rank and ties share their
  mean rank. Non-passing trajectories or groups with `|P| < 2` receive `1.0`.

  Args:
    ranks: Per-trajectory rank values (lower is better), shape `[B]`.
    passed_mask: Optional boolean mask of passing trajectories, shape `[B]`. If
      `None`, all trajectories are treated as passing.
    num_generations: Optional group size `G`. If `None`, treats input as one
      group of size `B`.
    rho: Rank discount slope in `[0, 1]`.

  Returns:
    Float32 array of quality scores `f_i` of shape `[B]`.
  """
  ranks_f32 = np.asarray(ranks, dtype=np.float32).reshape(-1)
  batch_size = ranks_f32.shape[0]
  if batch_size == 0:
    return np.empty(0, dtype=np.float32)

  if passed_mask is None:
    pass_bool = np.ones(batch_size, dtype=bool)
  else:
    pass_bool = np.asarray(passed_mask, dtype=bool).reshape(-1)

  group_size = batch_size if num_generations is None else num_generations
  num_groups = _validate_group_size(batch_size, group_size)
  qualities = np.ones(batch_size, dtype=np.float32)

  for g in range(num_groups):
    start = g * group_size
    end = start + group_size
    g_pass = pass_bool[start:end]
    pass_indices = np.flatnonzero(g_pass)
    num_pass = pass_indices.size
    if num_pass < 2:
      continue
    pass_ranks = _average_tie_ranks_1d(ranks_f32[start:end][pass_indices])
    f_vals = 1.0 - float(rho) * (pass_ranks - 1.0) / float(num_pass - 1)
    qualities[start + pass_indices] = f_vals.astype(np.float32)

  return qualities


def groupwise_advantage_redistribution(
    advantages: np.ndarray,
    passed: np.ndarray,
    quality: np.ndarray,
    num_generations: int,
    *,
    gar_lambda_max: float = 2.0,
    valid_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
  """Applies Groupwise Advantage Redistribution (Eq. GAR).

  Applied per group only when `|P| >= 2`, `A_j > 0` for all `j in P`, and
  `sum_{j in P} f_j A_j > 0`:
    `lambda = min(lambda_max, sum_{j in P} A_j / sum_{j in P} (f_j * A_j))`
    `A'_i = lambda * f_i * A_i` for `i in P`, else `A_i`
    `A^new_i = A'_i - mean_{j in V}(A'_j)`

  Args:
    advantages: Sequence-level normalized advantages `A_i`, shape `[B]`.
    passed: Boolean mask of passing trajectories, shape `[B]`.
    quality: Quality multipliers `f_i`, shape `[B]`.
    num_generations: Group size `G`.
    gar_lambda_max: Upper cap `lambda_max` on the redistribution multiplier.
    valid_mask: Optional boolean mask of valid trajectories, shape `[B]`.

  Returns:
    Tuple `(new_advantages, stats)` with `new_advantages` of shape `[B]`.
  """
  adv_f64 = np.asarray(advantages, dtype=np.float64).reshape(-1)
  pass_bool = np.asarray(passed, dtype=bool).reshape(-1)
  qual_f64 = np.asarray(quality, dtype=np.float64).reshape(-1)
  batch_size = adv_f64.shape[0]
  if batch_size == 0:
    empty_stats: dict[str, Any] = {
        "lambda": 1.0,
        "lambdas": [],
        "gar_applied_frac": 0.0,
        "advantage/lambda": 1.0,
        "advantage/gar_lambda_mean": 1.0,
        "advantage/gar_applied_frac": 0.0,
    }
    return adv_f64.astype(np.float32), empty_stats

  num_groups = _validate_group_size(batch_size, num_generations)
  if valid_mask is None:
    valid_bool = np.ones(batch_size, dtype=bool)
  else:
    valid_bool = np.asarray(valid_mask, dtype=bool).reshape(-1)

  out = adv_f64.astype(np.float32).copy()
  lambdas: list[float] = []
  applied_groups = 0

  for g in range(num_groups):
    start = g * num_generations
    end = start + num_generations
    g_valid = valid_bool[start:end]
    valid_idx = np.flatnonzero(g_valid)
    if valid_idx.size == 0:
      lambdas.append(1.0)
      out[start:end] = 0.0
      continue

    g_adv = adv_f64[start:end].copy()
    g_pass = pass_bool[start:end] & g_valid
    g_qual = qual_f64[start:end]
    pass_idx = np.flatnonzero(g_pass)

    lam = 1.0
    if pass_idx.size >= 2 and np.all(g_adv[pass_idx] > 0.0):
      weighted_sum = float(np.sum(g_qual[pass_idx] * g_adv[pass_idx]))
      if weighted_sum > 1e-12:
        raw_sum = float(np.sum(g_adv[pass_idx]))
        lam = min(float(gar_lambda_max), raw_sum / weighted_sum)
        g_adv[pass_idx] = lam * g_qual[pass_idx] * g_adv[pass_idx]
        mean_valid = float(np.mean(g_adv[valid_idx]))
        g_new = np.zeros(num_generations, dtype=np.float64)
        g_new[valid_idx] = g_adv[valid_idx] - mean_valid
        g_new_f32 = g_new.astype(np.float32)
        g_new_f32[valid_idx] -= np.float32(np.mean(g_new_f32[valid_idx]))
        out[start:end] = g_new_f32
        applied_groups += 1
      else:
        out[start:end] = np.where(g_valid, out[start:end], 0.0).astype(
            np.float32
        )
    else:
      out[start:end] = np.where(g_valid, out[start:end], 0.0).astype(
          np.float32
      )

    lambdas.append(lam)

  lambda_mean = float(np.mean(lambdas)) if lambdas else 1.0
  applied_frac = float(applied_groups) / float(max(num_groups, 1))
  stats: dict[str, Any] = {
      "lambda": lambda_mean,
      "lambdas": lambdas,
      "gar_applied_frac": applied_frac,
      "advantage/lambda": lambda_mean,
      "advantage/gar_lambda_mean": lambda_mean,
      "advantage/gar_applied_frac": applied_frac,
  }
  return out, stats


def span_level_token_advantages(
    seq_advantages: np.ndarray,
    token_flags: np.ndarray,
    completion_mask: np.ndarray,
    num_generations: int,
    *,
    span_kappa: float = 1.5,
    span_alpha_max: float = 2.0,
    span_beta_min: float = 0.5,
    span_beta_pos: float = 0.5,
    span_renormalize: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
  """Computes MiMo-V2-Pro span-level token advantages (Eq. SPAN).

  Args:
    seq_advantages: Sequence advantages `A^new_i` of shape `[B]`.
    token_flags: Binary redundant-token flags `h_{i,t}` of shape `[B, T]`.
    completion_mask: Active response token mask of shape `[B, T]`.
    num_generations: Group size `G`.
    span_kappa: Penalty amplification factor `kappa > 1` on redundant tokens of
      negative-advantage trajectories.
    span_alpha_max: Maximum boost `alpha_max` on clean tokens of
      positive-advantage trajectories.
    span_beta_min: Minimum retention `beta_min` on clean tokens of
      negative-advantage trajectories.
    span_beta_pos: Retention multiplier `beta_pos in [0, 1]` on flagged tokens
      of positive-advantage trajectories (`0.0` reproduces hard zeroing).
    span_renormalize: Whether to renormalize positive and negative token
      advantages so total group positive/negative advantage mass and per-sequence
      mean advantage are conserved after capping/flooring.

  Returns:
    Tuple `(token_advantages, stats)` where `token_advantages` has shape
    `[B, T]` and `dtype=float32`.
  """
  if not (0.0 <= span_beta_pos <= 1.0):
    raise ValueError("span_beta_pos must be in [0, 1].")
  seq_adv = np.asarray(seq_advantages, dtype=np.float64).reshape(-1)
  comp_mask = np.asarray(completion_mask) != 0
  red_mask = (np.asarray(token_flags) != 0) & comp_mask
  clean_mask = (~red_mask) & comp_mask

  batch_size, seq_len = comp_mask.shape
  if seq_adv.shape[0] != batch_size:
    raise ValueError(
        f"seq_advantages length {seq_adv.shape[0]} does not match"
        f" completion_mask batch size {batch_size}."
    )
  if batch_size == 0:
    empty_stats: dict[str, Any] = {
        "alpha": 1.0,
        "beta": 1.0,
        "alphas": [],
        "betas": [],
        "alpha_capped_frac": 0.0,
        "beta_floored_frac": 0.0,
        "mass_pos_before": 0.0,
        "mass_pos_after": 0.0,
        "mass_neg_before": 0.0,
        "mass_neg_after": 0.0,
        "mass_pos_ratio": 1.0,
        "mass_neg_ratio": 1.0,
        "advantage/alpha": 1.0,
        "advantage/beta": 1.0,
        "advantage/span_alpha_mean": 1.0,
        "advantage/span_beta_mean": 1.0,
        "advantage/alpha_capped_frac": 0.0,
        "advantage/beta_floored_frac": 0.0,
        "advantage/mass_pos_before": 0.0,
        "advantage/mass_pos_after": 0.0,
        "advantage/mass_neg_before": 0.0,
        "advantage/mass_neg_after": 0.0,
        "advantage/mass_pos_ratio": 1.0,
        "advantage/mass_neg_ratio": 1.0,
    }
    return np.zeros((0, seq_len), dtype=np.float32), empty_stats

  num_groups = _validate_group_size(batch_size, num_generations)
  out = np.zeros((batch_size, seq_len), dtype=np.float32)
  alphas: list[float] = []
  betas: list[float] = []
  alpha_capped = 0
  beta_floored = 0
  mass_pos_before_list: list[float] = []
  mass_pos_after_list: list[float] = []
  mass_neg_before_list: list[float] = []
  mass_neg_after_list: list[float] = []

  for g in range(num_groups):
    start = g * num_generations
    end = start + num_generations
    g_adv = seq_adv[start:end]  # [G]
    g_mask = comp_mask[start:end]  # [G, T]
    g_red = red_mask[start:end]  # [G, T]
    g_clean = clean_mask[start:end]  # [G, T]

    pos_traj = g_adv > 0.0
    neg_traj = g_adv < 0.0
    den_pos = 0.0
    den_neg = 0.0

    # Positive trajectories: boost clean tokens, attenuate redundant tokens by beta_pos.
    if np.any(pos_traj):
      pos_adv_col = g_adv[pos_traj, None]
      num_pos = float(np.sum(g_red[pos_traj] * pos_adv_col))
      den_pos = float(np.sum(g_clean[pos_traj] * pos_adv_col))
      if den_pos > 1e-12:
        raw_alpha = 1.0 + (1.0 - float(span_beta_pos)) * (num_pos / den_pos)
        if raw_alpha > span_alpha_max:
          alpha_capped += 1
        alpha = min(float(span_alpha_max), raw_alpha)
      else:
        alpha = 1.0
    else:
      alpha = 1.0

    # Negative trajectories: amplify redundant tokens by kappa, relax clean by beta.
    if np.any(neg_traj):
      neg_abs_col = np.abs(g_adv[neg_traj, None])
      num_neg = float(np.sum(g_red[neg_traj] * neg_abs_col))
      den_neg = float(np.sum(g_clean[neg_traj] * neg_abs_col))
      if den_neg > 1e-12:
        raw_beta = 1.0 - (float(span_kappa) - 1.0) * (num_neg / den_neg)
        if raw_beta < span_beta_min:
          beta_floored += 1
        beta = max(float(span_beta_min), raw_beta)
      else:
        beta = 1.0
    else:
      beta = 1.0

    alphas.append(alpha)
    betas.append(beta)

    g_out = np.zeros((num_generations, seq_len), dtype=np.float64)
    for i in range(num_generations):
      a_i = float(g_adv[i])
      if a_i > 0.0:
        if den_pos > 1e-12:
          weights = np.where(
              g_clean[i],
              alpha,
              np.where(g_red[i], float(span_beta_pos), 0.0),
          )
        else:
          fallback_w = 1.0 if span_renormalize else float(span_beta_pos)
          weights = np.where(g_mask[i], fallback_w, 0.0)
        g_out[i] = weights * a_i
      elif a_i < 0.0:
        if den_neg > 1e-12:
          weights = np.where(
              g_red[i],
              float(span_kappa),
              np.where(g_clean[i], beta, 0.0),
          )
        else:
          fallback_w = 1.0 if span_renormalize else float(span_kappa)
          weights = np.where(g_mask[i], fallback_w, 0.0)
        g_out[i] = weights * a_i

    pos_before = (
        float(np.sum(g_mask[pos_traj] * g_adv[pos_traj, None]))
        if np.any(pos_traj)
        else 0.0
    )
    neg_before = (
        float(np.sum(g_mask[neg_traj] * np.abs(g_adv[neg_traj, None])))
        if np.any(neg_traj)
        else 0.0
    )

    if span_renormalize:
      if np.any(pos_traj) and pos_before > 1e-12:
        pos_after_raw = float(np.sum(g_out[pos_traj]))
        if pos_after_raw > 1e-12:
          g_out[pos_traj] *= pos_before / pos_after_raw
        else:
          g_out[pos_traj] = np.where(
              g_mask[pos_traj], g_adv[pos_traj, None], 0.0
          )
      if np.any(neg_traj) and neg_before > 1e-12:
        neg_after_raw = float(np.sum(np.abs(g_out[neg_traj])))
        if neg_after_raw > 1e-12:
          g_out[neg_traj] *= neg_before / neg_after_raw
        else:
          g_out[neg_traj] = np.where(
              g_mask[neg_traj], g_adv[neg_traj, None], 0.0
          )

    out[start:end] = g_out.astype(np.float32)

    pos_after = (
        float(np.sum(out[start:end][pos_traj], dtype=np.float64))
        if np.any(pos_traj)
        else 0.0
    )
    neg_after = (
        float(np.sum(np.abs(out[start:end][neg_traj]), dtype=np.float64))
        if np.any(neg_traj)
        else 0.0
    )
    mass_pos_before_list.append(pos_before)
    mass_pos_after_list.append(pos_after)
    mass_neg_before_list.append(neg_before)
    mass_neg_after_list.append(neg_after)

  alpha_mean = float(np.mean(alphas)) if alphas else 1.0
  beta_mean = float(np.mean(betas)) if betas else 1.0
  alpha_cap_frac = float(alpha_capped) / float(max(num_groups, 1))
  beta_floor_frac = float(beta_floored) / float(max(num_groups, 1))
  total_pos_before = float(np.sum(mass_pos_before_list))
  total_pos_after = float(np.sum(mass_pos_after_list))
  total_neg_before = float(np.sum(mass_neg_before_list))
  total_neg_after = float(np.sum(mass_neg_after_list))
  mass_pos_ratio = (
      total_pos_after / total_pos_before if total_pos_before > 1e-12 else 1.0
  )
  mass_neg_ratio = (
      total_neg_after / total_neg_before if total_neg_before > 1e-12 else 1.0
  )
  stats: dict[str, Any] = {
      "alpha": alpha_mean,
      "beta": beta_mean,
      "alphas": alphas,
      "betas": betas,
      "alpha_capped_frac": alpha_cap_frac,
      "beta_floored_frac": beta_floor_frac,
      "mass_pos_before": total_pos_before,
      "mass_pos_after": total_pos_after,
      "mass_neg_before": total_neg_before,
      "mass_neg_after": total_neg_after,
      "mass_pos_ratio": mass_pos_ratio,
      "mass_neg_ratio": mass_neg_ratio,
      "advantage/alpha": alpha_mean,
      "advantage/beta": beta_mean,
      "advantage/span_alpha_mean": alpha_mean,
      "advantage/span_beta_mean": beta_mean,
      "advantage/alpha_capped_frac": alpha_cap_frac,
      "advantage/beta_floored_frac": beta_floor_frac,
      "advantage/mass_pos_before": total_pos_before,
      "advantage/mass_pos_after": total_pos_after,
      "advantage/mass_neg_before": total_neg_before,
      "advantage/mass_neg_after": total_neg_after,
      "advantage/mass_pos_ratio": mass_pos_ratio,
      "advantage/mass_neg_ratio": mass_neg_ratio,
  }
  return out, stats


def hard_mask_token_advantages(
    seq_advantages: np.ndarray,
    token_flags: np.ndarray,
    completion_mask: np.ndarray,
) -> np.ndarray:
  """Computes GSPO-MiMo hard-masked token advantages (Eq. MASK).

  Implements:
    `A_hat_{i,t} = (1 - h_{i,t}) * A_i * 1[t in T_i]`.

  Args:
    seq_advantages: Sequence advantages `[B]` (or token advantages `[B, T]`).
    token_flags: Binary redundant-token flags `h_{i,t}` of shape `[B, T]`.
    completion_mask: Active response token mask `[B, T]`.

  Returns:
    Float32 array of shape `[B, T]`.
  """
  comp_f32 = (np.asarray(completion_mask) != 0).astype(np.float32)
  clean_f32 = (np.asarray(token_flags) == 0).astype(np.float32) * comp_f32
  adv = np.asarray(seq_advantages, dtype=np.float32)
  if adv.ndim == 1:
    adv = adv[:, None]
  return (adv * clean_f32).astype(np.float32)


def _group_normalize_advantages(
    rewards: np.ndarray,
    num_generations: int,
    valid_mask: np.ndarray,
    *,
    std_normalize: bool = True,
) -> np.ndarray:
  """Computes group-normalized scalar advantages over valid trajectories."""
  batch_size = rewards.shape[0]
  if batch_size == 0:
    return np.empty(0, dtype=np.float32)
  num_groups = _validate_group_size(batch_size, num_generations)
  grouped_r = rewards.reshape(num_groups, num_generations).astype(np.float64)
  grouped_v = valid_mask.reshape(num_groups, num_generations)
  valid_counts = np.sum(grouped_v, axis=-1, keepdims=True)
  masked_sum = np.sum(np.where(grouped_v, grouped_r, 0.0), axis=-1, keepdims=True)
  mean_r = masked_sum / np.maximum(valid_counts, 1)
  centered = np.where(grouped_v, grouped_r - mean_r, 0.0)
  if std_normalize:
    sq_dev = np.sum(np.where(grouped_v, centered**2, 0.0), axis=-1, keepdims=True)
    std_r = np.sqrt(sq_dev / np.maximum(valid_counts - 1, 1))
    adv = centered / (std_r + 1e-6)
  else:
    adv = centered
  adv = np.where(grouped_v & (valid_counts >= 2), adv, 0.0)
  return adv.reshape(batch_size).astype(np.float32)


@function_registry.register_advantage_estimator("mimo")
def compute_mimo_advantages(
    rewards: np.ndarray,
    num_generations: int,
    valid_mask: np.ndarray | None = None,
    *,
    passed: np.ndarray | None = None,
    lengths: np.ndarray | None = None,
    quality: np.ndarray | None = None,
    ranks: np.ndarray | None = None,
    token_flags: np.ndarray | None = None,
    completion_mask: np.ndarray | None = None,
    cfg: algorithm_config.MimoAdvantageConfig | None = None,
    stats_out: dict[str, Any] | None = None,
    **_: Any,
) -> np.ndarray:
  """Registered `'mimo'` advantage estimator chaining LP -> Norm -> GAR -> SPAN.

  Args:
    rewards: Sequence rewards of shape `[B]`.
    num_generations: Group size `G`.
    valid_mask: Optional boolean mask of valid trajectories `[B]`.
    passed: Optional boolean mask of passing trajectories `[B]`.
    lengths: Optional per-trajectory response lengths `[B]`.
    quality: Optional precomputed quality multipliers `f_i` `[B]`.
    ranks: Optional per-trajectory group ranks `[B]`.
    token_flags: Optional binary redundant-token mask `h_{i,t}` `[B, T]`.
    completion_mask: Active response token mask of shape `[B, T]`.
    cfg: Optional `MimoAdvantageConfig` hyperparameters.
    stats_out: Optional mutable dictionary populated with LP/GAR/SPAN metrics.

  Returns:
    Token-level advantages `float32[B, T]`.
  """
  mimo_cfg = cfg if cfg is not None else algorithm_config.MimoAdvantageConfig()
  rewards_arr = np.asarray(rewards, dtype=np.float32).reshape(-1)
  batch_size = rewards_arr.shape[0]

  if completion_mask is None:
    if token_flags is not None:
      comp_mask = np.ones_like(np.asarray(token_flags), dtype=np.float32)
    else:
      comp_mask = np.ones((batch_size, 1), dtype=np.float32)
  else:
    comp_mask = np.asarray(completion_mask, dtype=np.float32)

  if valid_mask is None:
    valid_bool = np.ones(batch_size, dtype=bool)
  else:
    valid_bool = np.asarray(valid_mask, dtype=bool).reshape(-1)

  if passed is None:
    pass_bool = (rewards_arr > 0.0) & valid_bool
  else:
    pass_bool = np.asarray(passed, dtype=bool).reshape(-1) & valid_bool

  if lengths is None:
    lengths_arr = np.sum(comp_mask != 0, axis=-1).astype(np.int32)
  else:
    lengths_arr = np.asarray(lengths, dtype=np.int32).reshape(-1)

  if token_flags is None:
    red_mask = np.zeros_like(comp_mask, dtype=np.int8)
  else:
    red_mask = np.asarray(token_flags, dtype=np.int8)

  penalized_rewards, lp_stats = group_relative_length_penalty(
      rewards_arr,
      pass_bool,
      lengths_arr,
      num_generations,
      lp_pass_rate_threshold=mimo_cfg.lp_pass_rate_threshold,
      lp_max_penalty=mimo_cfg.lp_max_penalty,
      lp_dead_zone=mimo_cfg.lp_dead_zone,
      lp_saturation=mimo_cfg.lp_saturation,
      lp_gamma=mimo_cfg.lp_gamma,
      valid_mask=valid_bool,
  )

  seq_adv = _group_normalize_advantages(
      penalized_rewards,
      num_generations,
      valid_bool,
      std_normalize=mimo_cfg.std_normalize,
  )

  if quality is None:
    if ranks is not None:
      qual_arr = quality_from_ranks(
          ranks,
          pass_bool,
          num_generations=num_generations,
          rho=mimo_cfg.gar_rank_rho,
      )
    else:
      qual_arr = np.ones(batch_size, dtype=np.float32)
  else:
    qual_arr = np.asarray(quality, dtype=np.float32).reshape(-1)

  gar_adv, gar_stats = groupwise_advantage_redistribution(
      seq_adv,
      pass_bool,
      qual_arr,
      num_generations,
      gar_lambda_max=mimo_cfg.gar_lambda_max,
      valid_mask=valid_bool,
  )

  token_adv, span_stats = span_level_token_advantages(
      gar_adv,
      red_mask,
      comp_mask,
      num_generations,
      span_kappa=mimo_cfg.span_kappa,
      span_alpha_max=mimo_cfg.span_alpha_max,
      span_beta_min=mimo_cfg.span_beta_min,
      span_beta_pos=mimo_cfg.span_beta_pos,
      span_renormalize=mimo_cfg.span_renormalize,
  )

  token_adv = np.where(valid_bool[:, None], token_adv, 0.0).astype(np.float32)

  if stats_out is not None:
    stats_out.update(lp_stats)
    stats_out.update(gar_stats)
    stats_out.update(span_stats)

  return token_adv
