# Copyright 2026 Google LLC
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

"""Algorithm core implementations for RL and Agentic RL learners."""

import functools
from typing import Any, Callable
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from tunix.rl import advantages as rl_advantages
from tunix.rl import algorithm_config
from tunix.rl import common
from tunix.rl import function_registry
from tunix.sft import utils as sft_utils

registry = function_registry.default_registry

# ==============================================================================
# Utils
# ==============================================================================


@registry.register("advantage_estimator", "gae")
@jax.jit
def compute_gae_advantages(
    rewards: jax.Array,
    values: jax.Array,
    completion_mask: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
  """Compute advantages using Generalized Advantage Estimation (GAE).

  Computing GAE is a two-step process:

  First, compute the temporal difference (TF), `δ_t`, for each timestep `t`:

  ```
  δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
  ```

  Then, compute the GAE advantage, `A_t`, by summing the discounted TD
  residuals. It is calculated recursively, starting from the last timestep:

  ```
  A_t = δ_t + (γ * λ) * A_{t+1}
  ```

  where:

  - `A_t` is the GAE advantage at timestep `t`.
  - `δ_t` is the temporal difference at timestep `t`.
  - `γ` is the discount factor.
  - `λ` is the GAE lambda parameter.
  - `V(s_t)` is the value function at timestep `t`.
  - `r_t` is the reward at timestep `t`.

  Args:
    rewards: A 2D array of rewards for each step in the rollout.
    values: A 2D array of value estimates from the critic for each step.
    completion_mask: A 2D mask, which is 0 for padding tokens.
    gamma: The discount factor, `γ`.
    gae_lambda: The GAE lambda parameter, `λ`.

  Returns:
    A tuple of two 2D arrays - advantages and returns for each step.
  """
  batch_size = values.shape[0]

  def gae_step(state_t_plus_1, xs):
    # Unpack state and inputs.
    gae_t_plus_1, next_values = state_t_plus_1
    rewards_t, values_t, mask_t = xs

    # Compute Temporal Difference (TD).
    delta = rewards_t + gamma * next_values - values_t
    # Compute GAE for this time step.
    gae_t = delta + gamma * gae_lambda * gae_t_plus_1

    # Skip values on non-completion tokens.
    next_values = values_t * mask_t + (1 - mask_t) * next_values
    gae_t = gae_t * mask_t + (1 - mask_t) * gae_t_plus_1

    # New state to carry over comprises `gae_t` and `next_values`. Output for
    # this step is `gae_t`.
    return (gae_t, next_values), gae_t

  _, advantages_transposed = jax.lax.scan(
      gae_step,
      init=(jnp.zeros((batch_size,)), jnp.zeros((batch_size,))),
      xs=(
          jnp.transpose(jnp.array(rewards)),
          jnp.transpose(jnp.array(values)),
          jnp.transpose(jnp.array(completion_mask)),
      ),
      reverse=True,
  )
  advantages = jnp.transpose(advantages_transposed)
  returns = advantages + values

  # Normalise advantages.
  advantages = masked_whiten(advantages, completion_mask)
  return advantages, returns


@jax.jit
def masked_whiten(
    x: jax.Array,
    completion_mask: jax.Array,
) -> jax.Array:
  """Normalize the input array."""
  x_mean = masked_mean(x, completion_mask)
  x_var = masked_var(
      x,
      completion_mask,
      x_mean,
  )
  x = (x - x_mean) * jax.lax.rsqrt(x_var + 1e-8)
  return x


@functools.partial(jax.jit, static_argnames=("axis",))
def masked_mean(
    x: jax.Array, mask: jax.Array, axis: int | None = None
) -> jax.Array:
  """Compute the mean of a masked array."""
  cast_mask = mask.astype(x.dtype)
  safe_x = jnp.where(cast_mask > 0, x, 0.0)
  return jnp.sum(safe_x, axis=axis) / (
      jnp.sum(cast_mask, axis=axis) + 1e-8
  )


@jax.jit
def masked_var(
    x: jax.Array,
    mask: jax.Array,
    mean: jax.Array | None = None,
) -> jax.Array:
  """Compute the variance of a masked array."""
  cast_mask = mask.astype(x.dtype)
  if mean is None:
    mean = masked_mean(x, cast_mask)

  diff = jnp.where(cast_mask > 0, x - mean, 0.0)
  variance = masked_mean(jnp.square(diff), cast_mask)

  mask_sum = cast_mask.sum()
  bessel_corr = jnp.where(
      mask_sum > 1.0, mask_sum / jnp.maximum(mask_sum - 1.0, 1.0), 1.0
  )
  return variance * bessel_corr


# ==============================================================================
# PPO Core
# ==============================================================================


@function_registry.register_policy_loss_fn("ppo")
def ppo_policy_loss_fn(
    model,
    train_example,
    algo_config,
    pad_id,
    eos_id,
    **kwargs,
) -> sft_utils.LossOutput:
  """PPO policy loss function."""
  epsilon_low = algo_config.epsilon_low
  epsilon_high = algo_config.epsilon_high
  entropy_coef = algo_config.entropy_coef

  completion_ids = train_example.completion_ids
  completion_mask = train_example.completion_mask

  return_entropy = entropy_coef is not None and entropy_coef != 0.0
  graphdef, state = nnx.split(model)
  outputs = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
      return_entropy=return_entropy,
      temperature=getattr(algo_config, "temperature", None),
      segment_ids=getattr(train_example, "segment_ids", None),
      segment_positions=getattr(train_example, "segment_positions", None),
      chunk_size=kwargs.get("compute_logps_chunk_size", 0),
  )
  if return_entropy:
    per_token_logps, token_entropy = outputs
  else:
    per_token_logps = outputs

  old_per_token_logps = train_example.old_per_token_logps
  active_mask = (
      (completion_mask > 0)
      & jnp.isfinite(per_token_logps)
      & jnp.isfinite(old_per_token_logps)
  )
  safe_per_token_logps = jnp.where(active_mask, per_token_logps, 0.0)
  safe_old_per_token_logps = jnp.where(active_mask, old_per_token_logps, 0.0)
  raw_adv = (
      train_example.advantages
      if train_example.advantages.ndim == 2
      else jnp.expand_dims(train_example.advantages, -1)
  )
  advantages = jnp.where(
      active_mask & jnp.isfinite(raw_adv),
      raw_adv,
      0.0,
  )

  seq_importance_ratio = jnp.exp(
      jnp.clip(safe_per_token_logps - safe_old_per_token_logps, -20.0, 20.0)
  )

  # Compute pg_clipfrac
  pg_losses_1 = -seq_importance_ratio * advantages
  pg_losses_2 = (
      -jnp.clip(seq_importance_ratio, 1 - epsilon_low, 1 + epsilon_high)
      * advantages
  )

  per_token_loss = jnp.maximum(pg_losses_1, pg_losses_2)

  # add dual clip logic
  epsilon_c = getattr(algo_config, "epsilon_c", None)
  if epsilon_c is not None:
    pg_loss_3 = -epsilon_c * advantages
  else:
    pg_loss_3 = per_token_loss
  unreduced_pg_clipfrac_lower = jnp.sum(
      ((per_token_loss > pg_loss_3) & (advantages < 0.0)).astype(jnp.float32)
  )

  pg_loss_clipped_dual = jnp.minimum(pg_loss_3, per_token_loss)
  pg_losses = jnp.where(advantages < 0.0, pg_loss_clipped_dual, per_token_loss)

  denominator = jnp.sum(completion_mask)
  unreduced_pg_clipfrac = jnp.sum(
      jnp.greater(pg_losses_2, pg_losses_1).astype(jnp.float32)
  )
  unreduced_policy_loss = jnp.sum(pg_losses)

  aux = {
      "pg_clipfrac": sft_utils.WeightedMetric(
          unreduced_pg_clipfrac, denominator, min_denom=1.0
      ),
      "pg_clipfrac_lower": sft_utils.WeightedMetric(
          unreduced_pg_clipfrac_lower, denominator, min_denom=1.0
      ),
  }

  if return_entropy:
    unreduced_entropy = jnp.sum(
        jnp.where(active_mask, token_entropy, 0.0)  # pyrefly: ignore[unbound-name]
    )
    unreduced_policy_loss = (
        unreduced_policy_loss - entropy_coef * unreduced_entropy
    )
    aux["loss/entropy"] = sft_utils.WeightedMetric(
        unreduced_entropy, denominator, min_denom=1.0
    )

  # kl penalty term logic as before
  kl_coef = getattr(algo_config, "kl_coef", 0.0)
  if kl_coef > 0.0 and train_example.ref_per_token_logps is not None:
    kl = common.compute_kl_divergence(
        per_token_logps,
        train_example.ref_per_token_logps,
        "kl",
        clamp_value=getattr(algo_config, "kl_clamp_value", None),
    )
    unreduced_kl = jnp.sum(jnp.where(active_mask, kl, 0.0))
    unreduced_policy_loss = unreduced_policy_loss + kl_coef * unreduced_kl
    aux["kl"] = sft_utils.WeightedMetric(
        unreduced_kl, denominator, min_denom=1.0
    )

  return sft_utils.LossOutput(
      primary_loss=sft_utils.WeightedMetric(
          unreduced_policy_loss, denominator, min_denom=1.0
      ),
      aux_metrics=aux,
  )


@function_registry.register_value_loss_fn("ppo")
def ppo_value_loss_fn(
    model: nnx.Module,
    train_example,
    clip_range_value: float | None,
    pad_id: int,
    eos_id: int,
) -> sft_utils.LossOutput:
  """Computes the value loss for PPO."""

  prompt_ids, completion_ids, completion_mask = (
      train_example.prompt_ids,
      train_example.completion_ids,
      train_example.completion_mask,
  )
  # ====== Loss ======
  values = train_example.old_values
  returns = train_example.returns

  segment_ids = getattr(train_example, "segment_ids", None)
  if segment_ids is not None:
    # For packed sequences, prompt_ids is empty and completion_ids holds the
    # full sequence.
    # We predict values for token t using the model's output at t-1.
    logits_to_keep = completion_ids.shape[1] - 1
  else:
    logits_to_keep = completion_ids.shape[1]

  # Get new values.
  vpreds = common.compute_score(
      model,
      prompt_ids,
      completion_ids,
      pad_id,
      eos_id,
      stop_gradient=False,
      segment_ids=segment_ids,
      segment_positions=getattr(train_example, "segment_positions", None),
  )
  vpreds = vpreds[:, -logits_to_keep - 1 : -1]

  if segment_ids is not None:
    # Pad the first token's value with 0.0, since it has no preceding token to predict it.
    vpreds = jnp.pad(vpreds, ((0, 0), (1, 0)), constant_values=0.0)
  active_mask = (
      (completion_mask > 0)
      & jnp.isfinite(values)
      & jnp.isfinite(returns)
      & jnp.isfinite(vpreds)
  )
  safe_values = jnp.where(active_mask, values, 0.0)
  safe_returns = jnp.where(active_mask, returns, 0.0)
  safe_vpreds = jnp.where(active_mask, vpreds, 0.0)
  if clip_range_value is not None:
    vpred_clipped = jnp.clip(
        safe_vpreds,
        safe_values - clip_range_value,
        safe_values + clip_range_value,
    )
  else:
    vpred_clipped = safe_vpreds
  vf_losses1 = jnp.square(safe_vpreds - safe_returns)
  vf_losses2 = jnp.square(vpred_clipped - safe_returns)

  clipped_vf_losses = jnp.maximum(vf_losses1, vf_losses2)

  denominator = jnp.sum(completion_mask)
  unreduced_vf_loss = 0.5 * jnp.sum(clipped_vf_losses)
  unreduced_vpred_mean = jnp.sum(safe_vpreds)
  unreduced_vf_clipfrac = jnp.sum(
      jnp.greater(vf_losses2, vf_losses1).astype(jnp.float32)
  )
  unreduced_return_mean = jnp.sum(safe_returns)

  primary_loss = sft_utils.WeightedMetric(
      unreduced_vf_loss, denominator, min_denom=1.0
  )
  aux = {
      "vf_loss": primary_loss,
      "vpred_mean": sft_utils.WeightedMetric(
          unreduced_vpred_mean, denominator, min_denom=1.0
      ),
      "vf_clipfrac": sft_utils.WeightedMetric(
          unreduced_vf_clipfrac, denominator, min_denom=1.0
      ),
      "return_mean": sft_utils.WeightedMetric(
          unreduced_return_mean, denominator, min_denom=1.0
      ),
  }

  return sft_utils.LossOutput(primary_loss=primary_loss, aux_metrics=aux)


def compute_gspo_importance_ratio(
    safe_per_token_logps: jax.Array,
    safe_old_per_token_logps: jax.Array,
    completion_mask: jax.Array,
    *,
    loss_algo: str = "grpo",
    gspo_pool_mode: str = "sequence",
    pool_mode: str | None = None,
    turn_ids: jax.Array | None = None,
    num_turn_segments: int | None = None,
    token_flags: jax.Array | None = None,
    segment_ids: jax.Array | None = None,
    num_segments: int | None = None,
) -> tuple[jax.Array, jax.Array]:
  """Computes importance ratio and log ratio, applying GSPO pooling if configured.

  When `loss_algo` is `'gspo-token'` or `'gspo-turn'`, sequence-, turn-, span-,
  or sub-turn span-level (`'turn_span'`) geometric mean log-ratios are pooled
  and substituted into the ratio value using the stop-gradient trick to preserve
  per-token policy gradients:
    `ratio = exp(clip(logp - stop_grad(logp) + stop_grad(pooled_mean)))`.
  Otherwise, returns standard per-token `(exp(clip(diff)), clip(diff))`.

  Args:
    safe_per_token_logps: Current policy per-token log-probabilities `[B, T]`.
    safe_old_per_token_logps: Rollout policy per-token log-probabilities `[B, T]`.
    completion_mask: Mask of active assistant tokens `[B, T]`.
    loss_algo: Algorithm name (`'grpo'`, `'gspo-token'`, or `'gspo-turn'`).
    gspo_pool_mode: Pooling granularity (`'sequence'`, `'turn'`, `'span'`, or
      `'turn_span'`).
    pool_mode: Optional alias for `gspo_pool_mode`.
    turn_ids: Optional assistant turn index array `[B, T]`.
    num_turn_segments: Optional static max count of turn segments.
    token_flags: Optional binary fault flag array `[B, T]`.
    segment_ids: Optional packed sequence segment index array `[B, T]`.
    num_segments: Optional static max count of packed segments.

  Returns:
    Tuple `(is_ratio, seq_importance_ratio)` where `is_ratio` is `[B, T]` and
    `seq_importance_ratio` is the log ratio `[B, T]`.
  """
  if pool_mode is not None:
    gspo_pool_mode = pool_mode
  log_diff = safe_per_token_logps - safe_old_per_token_logps
  seq_importance_ratio = jnp.clip(log_diff, max=20.0, min=-20.0)

  if loss_algo in ("gspo-token", "gspo-turn"):
    effective_pool_mode = gspo_pool_mode
    if loss_algo == "gspo-turn" and effective_pool_mode == "sequence":
      effective_pool_mode = "turn"

    if effective_pool_mode == "turn_span":
      if turn_ids is not None and token_flags is not None:
        pool_seg_ids = jnp.where(
            (completion_mask > 0) & (turn_ids > 0),
            turn_ids.astype(jnp.int32) * 2
            + (token_flags > 0).astype(jnp.int32),
            0,
        )
        base_num_segs = int(num_turn_segments or num_segments or 8)
        pool_num_segs = base_num_segs * 2 + 2
      elif turn_ids is not None:
        pool_seg_ids = turn_ids
        pool_num_segs = num_turn_segments or num_segments or 8
      elif token_flags is not None:
        pool_seg_ids = jnp.where(
            completion_mask > 0,
            1 + (token_flags > 0).astype(jnp.int32),
            0,
        )
        pool_num_segs = 4
      elif segment_ids is not None:
        pool_seg_ids = segment_ids
        pool_num_segs = num_segments or num_turn_segments or 8
      else:
        pool_seg_ids = None
        pool_num_segs = None
    elif effective_pool_mode == "span":
      if token_flags is not None:
        if segment_ids is not None:
          pool_seg_ids = jnp.where(
              (completion_mask > 0) & (segment_ids > 0),
              segment_ids.astype(jnp.int32) * 2
              + (token_flags > 0).astype(jnp.int32),
              0,
          )
          base_num_segs = int(num_segments or num_turn_segments or 8)
          pool_num_segs = base_num_segs * 2 + 2
        else:
          pool_seg_ids = jnp.where(
              completion_mask > 0,
              1 + (token_flags > 0).astype(jnp.int32),
              0,
          )
          pool_num_segs = 4
      elif segment_ids is not None:
        pool_seg_ids = segment_ids
        pool_num_segs = num_segments or num_turn_segments or 8
      else:
        pool_seg_ids = None
        pool_num_segs = None
    elif effective_pool_mode == "turn" and turn_ids is not None:
      pool_seg_ids = turn_ids
      pool_num_segs = num_turn_segments or num_segments or 8
    elif segment_ids is not None:
      pool_seg_ids = segment_ids
      pool_num_segs = num_segments or num_turn_segments or 8
    else:
      pool_seg_ids = None
      pool_num_segs = None

    if pool_seg_ids is not None and pool_num_segs is not None:
      if isinstance(pool_seg_ids, np.ndarray):
        pool_num_segs = max(int(pool_num_segs), int(np.max(pool_seg_ids)) + 1)
      elif not isinstance(pool_seg_ids, jax.core.Tracer):
        max_seg = jnp.max(pool_seg_ids)
        if not isinstance(max_seg, jax.core.Tracer):
          pool_num_segs = max(int(pool_num_segs), int(max_seg) + 1)

    if pool_seg_ids is None:
      seq_mean_ratio = jnp.sum(seq_importance_ratio, axis=-1) / jnp.clip(
          completion_mask.sum(-1), min=1
      )
      seq_mean_ratio = jnp.expand_dims(seq_mean_ratio, axis=-1)
    else:
      valid_seg_mask = completion_mask * (pool_seg_ids > 0).astype(
          completion_mask.dtype
      )
      per_seg_sum = common.segmented_sum(
          jnp.where(valid_seg_mask > 0, seq_importance_ratio, 0.0),
          pool_seg_ids,
          pool_num_segs,  # pyrefly: ignore[bad-argument-type]
      )
      per_seg_count = common.segmented_count(
          pool_seg_ids,
          pool_num_segs,  # pyrefly: ignore[bad-argument-type]
          mask=valid_seg_mask,
      )
      per_seg_mean = per_seg_sum / jnp.clip(per_seg_count, min=1.0)
      seq_mean_ratio = jnp.take_along_axis(
          per_seg_mean, pool_seg_ids.astype(jnp.int32), axis=1
      )
      seq_mean_ratio = jnp.where(valid_seg_mask > 0, seq_mean_ratio, 0.0)

    seq_importance_ratio = (
        safe_per_token_logps
        - jax.lax.stop_gradient(safe_per_token_logps)
        + jax.lax.stop_gradient(seq_mean_ratio)
    )
    seq_importance_ratio = jnp.clip(seq_importance_ratio, min=-10.0, max=10.0)

  is_ratio = jnp.exp(seq_importance_ratio)
  return is_ratio, seq_importance_ratio


def compute_policy_loss_and_clipping(
    adv: jax.Array,
    is_ratio: jax.Array,
    active_mask: jax.Array,
    *,
    clip_mode: str = "ppo",
    epsilon: float = 0.2,
    epsilon_high: float = 0.2,
    epsilon_neg: float | None = None,
    epsilon_neg_high: float | None = None,
    epsilon_c: float | None = None,
    completion_mask: jax.Array | None = None,
    loss_aggregation_mode: str = "token-mean",
    segment_ids: jax.Array | None = None,
    num_segments: int | None = None,
    gspo_length_adaptive_clipping: bool = False,
    gspo_ref_length: int = 1024,
    gspo_clip_min: float = 0.001,
    gspo_clip_max: float = 0.05,
    turn_ids: jax.Array | None = None,
    num_turn_segments: int | None = None,
    gspo_pool_mode: str = "sequence",
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, sft_utils.WeightedMetric
]:
  """Computes policy loss, individual branch losses, and clipping metrics.

  Supports both PPO dual clipping and sign-decoupled hard mask clipping
  (`clip_mode="mask"`), with optional length-adaptive trust-region scaling.

  Args:
    adv: Advantage array `[B, T]`.
    is_ratio: Importance ratio array `[B, T]`.
    active_mask: Boolean active token mask `[B, T]`.
    clip_mode: Clipping mechanism (`'ppo'` or `'mask'`).
    epsilon: Positive advantage lower clip bound `1 - eps`.
    epsilon_high: Positive advantage upper clip bound `1 + eps_high`.
    epsilon_neg: Negative advantage lower clip bound `1 - eps_neg`.
    epsilon_neg_high: Negative advantage upper clip bound `1 + eps_neg_high`.
    epsilon_c: Dual-clip PPO lower bound parameter.
    completion_mask: Mask of active assistant tokens `[B, T]`.
    loss_aggregation_mode: Aggregation mode for loss metrics.
    segment_ids: Optional sequence segment IDs.
    num_segments: Optional sequence segment count.
    gspo_length_adaptive_clipping: Whether to scale epsilon by
      `sqrt(gspo_ref_length / L)`.
    gspo_ref_length: Reference length `L_ref` for length-adaptive clipping.
    gspo_clip_min: Minimum bound for length-scaled epsilon.
    gspo_clip_max: Maximum bound for length-scaled epsilon.
    turn_ids: Optional turn IDs `[B, T]` for turn-level length scaling.
    num_turn_segments: Optional static max count of turn segments.
    gspo_pool_mode: GSPO pooling granularity.

  Returns:
    Tuple `(per_token_loss, pg_loss_1, pg_loss_2, unreduced_clip_frac, pg_clipfrac_lower)`.
  """
  raw_eps_neg = epsilon_neg if epsilon_neg is not None else epsilon
  raw_eps_neg_high = (
      epsilon_neg_high if epsilon_neg_high is not None else epsilon_high
  )
  if gspo_length_adaptive_clipping:
    mask_for_len = (
        completion_mask
        if completion_mask is not None
        else active_mask.astype(jnp.float32)
    )
    if turn_ids is not None and "turn" in gspo_pool_mode:
      t_num_segs = int(num_turn_segments or num_segments or 8)
      if isinstance(turn_ids, np.ndarray):
        t_num_segs = max(t_num_segs, int(np.max(turn_ids)) + 1)
      elif not isinstance(turn_ids, jax.core.Tracer):
        max_t = jnp.max(turn_ids)
        if not isinstance(max_t, jax.core.Tracer):
          t_num_segs = max(t_num_segs, int(max_t) + 1)
      valid_t_mask = ((mask_for_len > 0) & (turn_ids > 0)).astype(jnp.float32)
      per_turn_counts = common.segmented_count(
          turn_ids, t_num_segs, mask=valid_t_mask
      )
      eff_len = jnp.take_along_axis(
          per_turn_counts, turn_ids.astype(jnp.int32), axis=1
      )
    elif segment_ids is not None and num_segments is not None:
      per_seg_counts = common.segmented_count(
          segment_ids,
          num_segments,
          mask=(mask_for_len > 0).astype(jnp.float32),
      )
      eff_len = jnp.take_along_axis(
          per_seg_counts, segment_ids.astype(jnp.int32), axis=1
      )
    else:
      eff_len = jnp.sum(
          (mask_for_len > 0).astype(jnp.float32), axis=-1, keepdims=True
      )
    len_scale = jnp.sqrt(
        jnp.float32(gspo_ref_length) / jnp.maximum(eff_len, 1.0)
    )
    eff_eps = jnp.clip(epsilon * len_scale, gspo_clip_min, gspo_clip_max)
    eff_eps_high = jnp.clip(
        epsilon_high * len_scale, gspo_clip_min, gspo_clip_max
    )
    eff_eps_neg = jnp.clip(
        raw_eps_neg * len_scale, gspo_clip_min, gspo_clip_max
    )
    eff_eps_neg_high = jnp.clip(
        raw_eps_neg_high * len_scale, gspo_clip_min, gspo_clip_max
    )
  else:
    eff_eps = epsilon
    eff_eps_high = epsilon_high
    eff_eps_neg = raw_eps_neg
    eff_eps_neg_high = raw_eps_neg_high

  if clip_mode == "mask":
    lo = jnp.where(adv >= 0.0, 1.0 - eff_eps, 1.0 - eff_eps_neg)
    hi = jnp.where(adv >= 0.0, 1.0 + eff_eps_high, 1.0 + eff_eps_neg_high)
    keep = jax.lax.stop_gradient(
        ((is_ratio >= lo) & (is_ratio <= hi)).astype(jnp.float32)
    )
    per_token_loss = jnp.where(active_mask, -adv * is_ratio * keep, 0.0).astype(
        jnp.float32
    )
    pg_loss_1 = -adv * is_ratio
    pg_loss_2 = per_token_loss
    unreduced_clip_frac = jnp.sum(jnp.where(active_mask, 1.0 - keep, 0.0))
    pg_clipfrac_lower = sft_utils.WeightedMetric(
        jnp.array(0.0, dtype=jnp.float32), jnp.array(1.0, dtype=jnp.float32)
    )
  else:
    pg_loss_1 = -adv * is_ratio
    pg_loss_2 = -adv * jnp.clip(is_ratio, 1.0 - eff_eps, 1.0 + eff_eps_high)

    per_token_loss = jnp.maximum(pg_loss_1, pg_loss_2).astype(jnp.float32)

    unreduced_clip_frac = jnp.sum(
        jnp.greater(pg_loss_2, pg_loss_1).astype(jnp.float32)
    )

    if epsilon_c is not None:
      pg_loss_3 = -epsilon_c * adv
    else:
      pg_loss_3 = per_token_loss

    per_token_pg_clipfrac_lower = (
        (per_token_loss > pg_loss_3) & (adv < 0.0)
    ).astype(jnp.float32)
    pg_clipfrac_lower = common.aggregate_loss(
        per_token_pg_clipfrac_lower,
        completion_mask,
        loss_aggregation_mode,
        segment_ids=segment_ids,
        num_segments=num_segments,
    )

    pg_loss_clipped_dual = jnp.minimum(pg_loss_3, per_token_loss)
    per_token_loss = jnp.where(adv < 0.0, pg_loss_clipped_dual, per_token_loss)

  return (
      per_token_loss,
      pg_loss_1,
      pg_loss_2,
      unreduced_clip_frac,
      pg_clipfrac_lower,
  )


@function_registry.register_policy_loss_fn("grpo")
def grpo_loss_fn(
    model: nnx.Module,
    train_example: common.TrainExample,
    algo_config: algorithm_config.GRPOConfig,
    pad_id: int,
    eos_id: int,
    *,
    compute_logps_chunk_size: int = 0,
) -> sft_utils.LossOutput:
  """GRPO loss function.

  The loss aims to maximize the expected advantage of the chosen actions while
  constraining the policy updates to stay within a certain range of the
  reference policy.

  Args:
    model: The policy model to be trained.
    train_example: A `TrainExample` instance containing the processed input
      data, including prompt IDs, completion IDs, masks, advantages, and
      per-token log probabilities from the reference and policy models.
    algo_config: The algorithm config.
    pad_id: The pad ID from tokenizer.
    eos_id: The eos ID from tokenizer.
    compute_logps_chunk_size: Optional chunk size for log-prob computation.

  Returns:
    A LossOutput containing the loss and an aux dictionary.
  """
  beta = algo_config.beta
  epsilon = algo_config.epsilon
  loss_algo = algo_config.loss_algo
  clip_mode = algo_config.clip_mode
  epsilon_high = (
      algo_config.epsilon_high
      if algo_config.epsilon_high is not None
      else epsilon
  )
  epsilon_c = algo_config.epsilon_c
  loss_aggregation_mode = algo_config.loss_agg_mode

  completion_ids, completion_mask = (
      train_example.completion_ids,
      train_example.completion_mask,
  )
  # Packing metadata: `segment_ids` labels each token's sequence within a packed
  # row; `num_segments` is the static segment-bucket count. Both are None when
  # not packing, in which case every aggregate_loss/reduced_loss_agg below (and
  # the gspo-token pooling) takes its per-row branch unchanged.
  raw_segment_ids = train_example.segment_ids
  raw_num_segments = train_example.num_segments
  segment_positions = train_example.segment_positions
  is_packed = (raw_segment_ids is not None) and (segment_positions is not None)
  segment_ids = raw_segment_ids if is_packed else None
  num_segments = raw_num_segments if is_packed else None
  if isinstance(train_example, common.TrainExample):
    turn_ids = train_example.turn_ids
    num_turn_segments = train_example.num_turn_segments
    token_flags = train_example.token_flags
    token_mask = train_example.token_mask
    completion_attention_mask = train_example.completion_attention_mask
  else:
    turn_ids = None
    num_turn_segments = None
    token_flags = None
    token_mask = None
    completion_attention_mask = None

  # TODO(tsbao): split can be avoided with updated peft_trainer model handling.
  graphdef, state = nnx.split(model)
  if not isinstance(token_mask, (jax.Array, np.ndarray)):
    token_mask = None
  if token_mask is None and not is_packed:
    if (
        isinstance(completion_attention_mask, (jax.Array, np.ndarray))
        and train_example.prompt_mask is not None
    ):
      token_mask = jnp.concatenate(
          [train_example.prompt_mask, completion_attention_mask], axis=1
      )
  per_token_logps, token_entropy = common.compute_per_token_logps(
      graphdef,
      state,
      prompt_tokens=train_example.prompt_ids,
      completion_tokens=completion_ids,
      pad_id=pad_id,
      eos_id=eos_id,
      stop_gradient=False,
      return_entropy=True,
      segment_ids=segment_ids,
      segment_positions=segment_positions if is_packed else None,
      temperature=algo_config.temperature,
      chunk_size=compute_logps_chunk_size,
      routed_experts=train_example.routed_experts,
      token_mask=token_mask,
  )
  per_token_logps = jnp.astype(per_token_logps, jnp.float32)
  # TODO(tsbao): We should handle token level advantages.
  advantages = jnp.astype(train_example.advantages, jnp.float32)

  sampler_is_weights = train_example.sampler_is_weights
  sa_metrics = {}
  should_fuse_sampler_agreement = (
      algo_config.use_rollout_logps
      and train_example.old_per_token_logps is not None
      and sampler_is_weights is None
      and not train_example.sampler_agreement_applied
  )
  if should_fuse_sampler_agreement:
    rollout_per_token_logps = jnp.astype(
        train_example.old_per_token_logps, jnp.float32
    )
    trainer_per_token_logps = jax.lax.stop_gradient(per_token_logps)
    sa_metrics, computed_is_weights, filtered_completion_mask = (
        common.compute_sampler_trainer_agreement_jax(
            rollout_per_token_logps,
            trainer_per_token_logps,
            completion_mask,
            sampler_is=algo_config.sampler_is,
            sampler_is_threshold=algo_config.sampler_is_threshold,
            sampler_rs=algo_config.sampler_rs,
            sampler_rs_min=algo_config.sampler_rs_min,
            sampler_rs_max=algo_config.sampler_rs_max,
            seq_logprob_error_threshold=algo_config.seq_logprob_error_threshold,
            segment_ids=segment_ids,
            num_segments=num_segments,
        )
    )
    if algo_config.seq_logprob_error_threshold is not None:
      completion_mask = filtered_completion_mask
    if computed_is_weights is not None:
      sampler_is_weights = computed_is_weights

  # Use on-policy stop_gradient(per_token_logps) as the PPO ratio baseline when:
  # 1. No rollout/old logps were provided, or
  # 2. Fused in-loss IS/RS/error-masking is active (since sampler_is_weights
  #    already corrects for trainer-vs-sampler divergence outside the PPO clip).
  use_on_policy_old_logps = (
      train_example.old_per_token_logps is None
      or (
          should_fuse_sampler_agreement
          and (
              algo_config.sampler_is is not None
              or algo_config.sampler_rs is not None
              or algo_config.seq_logprob_error_threshold is not None
          )
      )
  )
  if use_on_policy_old_logps:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
  else:
    old_per_token_logps = jnp.astype(
        train_example.old_per_token_logps, jnp.float32
    )

  active_mask = (
      (completion_mask > 0)
      & jnp.isfinite(per_token_logps)
      & jnp.isfinite(old_per_token_logps)
  )
  safe_per_token_logps = jnp.where(active_mask, per_token_logps, 0.0)
  safe_old_per_token_logps = jnp.where(active_mask, old_per_token_logps, 0.0)
  seq_importance_ratio = safe_per_token_logps - safe_old_per_token_logps
  # Record KL divergence before clipping.
  token_denom = jnp.sum(completion_mask)
  unreduced_ppo_kl = jnp.sum(-seq_importance_ratio)

  gspo_pool_mode = algo_config.gspo_pool_mode
  is_ratio, seq_importance_ratio = compute_gspo_importance_ratio(
      safe_per_token_logps,
      safe_old_per_token_logps,
      completion_mask,
      loss_algo=loss_algo,
      gspo_pool_mode=gspo_pool_mode,
      turn_ids=turn_ids,
      num_turn_segments=num_turn_segments,
      token_flags=token_flags,
      segment_ids=raw_segment_ids,
      num_segments=raw_num_segments,
  )

  # Advantages must be broadcast against seq_length.
  # When sequence packing is used, advantages are already 2D [B, seq_length].
  # When unpacked, they are 1D [B].
  raw_adv = (
      advantages if advantages.ndim == 2 else jnp.expand_dims(advantages, 1)
  )
  adv = jnp.where(active_mask & jnp.isfinite(raw_adv), raw_adv, 0.0)

  (
      per_token_loss,
      pg_loss_1,
      pg_loss_2,
      unreduced_clip_frac,
      pg_clipfrac_lower,
  ) = compute_policy_loss_and_clipping(
      adv,
      is_ratio,
      active_mask,
      clip_mode=clip_mode,
      epsilon=epsilon,
      epsilon_high=epsilon_high,
      epsilon_neg=algo_config.epsilon_neg,
      epsilon_neg_high=algo_config.epsilon_neg_high,
      epsilon_c=epsilon_c,
      completion_mask=completion_mask,
      loss_aggregation_mode=loss_aggregation_mode,
      segment_ids=segment_ids,
      num_segments=num_segments,
      gspo_length_adaptive_clipping=algo_config.gspo_length_adaptive_clipping,
      gspo_ref_length=algo_config.gspo_ref_length,
      gspo_clip_min=algo_config.gspo_clip_min,
      gspo_clip_max=algo_config.gspo_clip_max,
      turn_ids=turn_ids,
      num_turn_segments=num_turn_segments,
      gspo_pool_mode=gspo_pool_mode,
  )

  # Optional truncated importance-sampling (TIS) and/or rejection-sampling (RS)
  # correction for the residual sampler-vs-trainer log-probability mismatch.
  # The weights are precomputed upstream or computed in-loss above (detached)
  # and applied per token BEFORE loss aggregation so they affect the gradient
  # through the loss magnitude only, not as a stop-gradient bias on the ratio.
  if sampler_is_weights is not None:
    safe_sis = jnp.where(
        active_mask & jnp.isfinite(sampler_is_weights),
        sampler_is_weights.astype(jnp.float32),
        0.0,
    )
    per_token_loss = per_token_loss * safe_sis

  # Two independent aggregations of the same policy loss (equal today):
  #   unreduced (sum/denom, deferred) — feeds the gradient
  #   reduced   (eager per-sequence mean, pre-CL form) — metric only
  unreduced_pg_loss = common.aggregate_loss(
      per_token_loss,
      completion_mask,
      loss_aggregation_mode,
      segment_ids=segment_ids,
      num_segments=num_segments,
      min_seq_len_norm=algo_config.min_seq_len_norm,
  )
  reduced_pg_loss = common.reduced_loss_agg(
      per_token_loss,
      completion_mask,
      loss_aggregation_mode,
      segment_ids=segment_ids,
      num_segments=num_segments,
      min_seq_len_norm=algo_config.min_seq_len_norm,
  )
  total_loss = (
      unreduced_pg_loss  # KL added below when beta != 0; feeds gradient
  )
  # Per-token diagnostics — log only over assistant tokens (completion_mask).
  has_valid = jnp.any(active_mask)
  is_ratio_mean = masked_mean(is_ratio, active_mask)
  is_ratio_max = jnp.max(jnp.where(active_mask, is_ratio, 0.0))
  is_ratio_min = jnp.where(
      has_valid,
      jnp.min(jnp.where(active_mask, is_ratio, jnp.inf)),
      0.0,
  )
  log_ratio_mean = masked_mean(seq_importance_ratio, active_mask)
  log_ratio_abs_mean = masked_mean(jnp.abs(seq_importance_ratio), active_mask)
  seq_clipfrac = masked_mean(
      ((is_ratio < 1.0 - epsilon) | (is_ratio > 1.0 + epsilon_high)).astype(
          jnp.float32
      ),
      active_mask,
  )
  pg_loss_1_mean = masked_mean(pg_loss_1, active_mask)
  pg_loss_2_mean = masked_mean(pg_loss_2, active_mask)
  adv_abs_mean = masked_mean(jnp.abs(adv), active_mask)
  adv_max = jnp.where(
      has_valid,
      jnp.max(jnp.where(active_mask, adv, -jnp.inf)),
      0.0,
  )
  adv_min = jnp.where(
      has_valid,
      jnp.min(jnp.where(active_mask, adv, jnp.inf)),
      0.0,
  )
  nonzero_adv_frac = masked_mean(
      (jnp.abs(adv) > 1e-8).astype(jnp.float32), active_mask
  )
  aux: dict[str, jax.Array | sft_utils.WeightedMetric] = {
      "kl": sft_utils.WeightedMetric(jnp.array(0.0), jnp.array(1.0)),
      "kl_loss": sft_utils.WeightedMetric(jnp.array(0.0), jnp.array(1.0)),
      "reduced_pg_loss": reduced_pg_loss,
      # TODO(yuxzhang): equal to reduced_pg_loss today; diverges once sequence
      # packing lands (reduced -> segment-aware metric; unreduced -> global).
      "unreduced_pg_loss": unreduced_pg_loss,
      "pg_clipfrac": sft_utils.WeightedMetric(
          unreduced_clip_frac, token_denom, min_denom=1.0
      ),
      "seq_clipfrac": seq_clipfrac,
      "ppo_kl": sft_utils.WeightedMetric(
          unreduced_ppo_kl, token_denom, min_denom=1.0
      ),
      "pg_clipfrac_lower": pg_clipfrac_lower,
      "is_ratio/mean": is_ratio_mean,
      "is_ratio/max": is_ratio_max,
      "is_ratio/min": is_ratio_min,
      "log_ratio/mean": log_ratio_mean,
      "log_ratio_mean": log_ratio_mean,
      "log_ratio/abs_mean": log_ratio_abs_mean,
      "log_ratio_abs_mean": log_ratio_abs_mean,
      "pg_loss/unclipped_mean": pg_loss_1_mean,
      "pg_loss/clipped_mean": pg_loss_2_mean,
      "advantage/abs_mean": adv_abs_mean,
      "advantage/max": adv_max,
      "advantage/min": adv_min,
      "advantage/nonzero_frac": nonzero_adv_frac,
  }
  if sampler_is_weights is not None:
    aux["sampler_is/weight_mean"] = sft_utils.WeightedMetric(
        jnp.sum(safe_sis), token_denom, min_denom=1.0  # pyrefly: ignore[unbound-name]
    )
    aux["sampler_is/weight_min"] = jnp.where(
        has_valid,
        jnp.min(jnp.where(active_mask, safe_sis, jnp.inf)),  # pyrefly: ignore[unbound-name]
        0.0,
    )
  else:
    aux["sampler_is/weight_mean"] = sft_utils.WeightedMetric(
        token_denom, token_denom, min_denom=1.0
    )
    aux["sampler_is/weight_min"] = jnp.float32(1.0)
  for metric_name, (metric_val, _) in sa_metrics.items():
    aux[metric_name] = metric_val
  # We do not always compute KL divergence (e.g. when beta is 0.0 unless
  # force_compute_kl is True).
  if train_example.ref_per_token_logps is not None:
    kl = common.compute_kl_divergence(
        per_token_logps,
        train_example.ref_per_token_logps,
        algo_config.kl_loss_mode,
        clamp_value=algo_config.kl_clamp_value,
    )
    kl = jnp.where(active_mask, kl, 0.0)
    unreduced_kl = jnp.astype(jnp.sum(kl), jnp.float32)
    aux["kl"] = sft_utils.WeightedMetric(
        unreduced_kl, token_denom, min_denom=1.0
    )
    kl_loss = common.aggregate_loss(
        kl,
        completion_mask,
        loss_aggregation_mode,
        segment_ids=segment_ids,
        num_segments=num_segments,
    )
    aux["kl_loss"] = kl_loss
  if beta is not None and beta != 0.0:
    total_loss = sft_utils.WeightedMetric(
        unreduced_pg_loss.unreduced_sum + beta * kl_loss.unreduced_sum,  # pyrefly: ignore[unbound-name]
        unreduced_pg_loss.denominator,
        eps=unreduced_pg_loss.eps,
        min_denom=unreduced_pg_loss.min_denom,
    )

  entropy_loss = common.aggregate_loss(
      jnp.where(active_mask, token_entropy, 0.0),
      completion_mask,
      loss_aggregation_mode,
      segment_ids=segment_ids,
      num_segments=num_segments,
  )
  aux["entropy"] = entropy_loss

  return sft_utils.LossOutput(primary_loss=total_loss, aux_metrics=aux)


MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE = (
    rl_advantages.MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE
)
compute_advantages = rl_advantages.compute_advantages
compute_grpo_loo_advantages = rl_advantages.compute_grpo_loo_advantages
compute_rloo_advantages = rl_advantages.compute_rloo_advantages
compute_drgrpo_advantages = rl_advantages.compute_drgrpo_advantages


# ==============================================================================
# Trainer Model Input Builders
# ==============================================================================


def _algo_model_input(
    batch: Any,
    algo_config: algorithm_config.AlgorithmConfig,
    pad_id: int,
    eos_id: int,
) -> dict[str, Any]:
  return {
      "train_example": batch,
      "algo_config": algo_config,
      "pad_id": pad_id,
      "eos_id": eos_id,
  }


def _critic_model_input(
    batch: Any,
    *,
    clip_range_value: float | None,
    pad_id: int,
    eos_id: int,
) -> dict[str, Any]:
  return {
      "train_example": batch,
      "clip_range_value": clip_range_value,
      "pad_id": pad_id,
      "eos_id": eos_id,
  }


def build_gen_model_input_fn(
    algo_config: algorithm_config.AlgorithmConfig,
    *,
    pad_id: int,
    eos_id: int,
) -> Callable[[Any], dict[str, Any]]:
  """Returns a closure mapping a `TrainExample` batch to policy-loss kwargs.

  The trainer worker calls the returned function on every microbatch and
  forwards the resulting kwargs to the policy loss registered for
  `algo_config.policy_loss_fn`.

  Args:
    algo_config: Algorithm configuration; its `temperature` must already be set
      (`StandardRLProgram` propagates it from `generation_args`).
    pad_id: Pad token id handed to the loss function.
    eos_id: End-of-sequence token id handed to the loss function.
  """
  if algo_config.temperature is None:
    raise ValueError(
        "Trainer temperature is unset on algo_config. Configure temperature"
        " via generation_args on StandardRLProgram (which propagates it to"
        " algo_config.temperature) before building the trainer model input fn."
    )
  return functools.partial(
      _algo_model_input,
      algo_config=algo_config,
      pad_id=pad_id,
      eos_id=eos_id,
  )


def build_critic_model_input_fn(
    algo_config: algorithm_config.PPOConfig,
    *,
    pad_id: int,
    eos_id: int,
) -> Callable[[Any], dict[str, Any]]:
  """Returns a closure mapping a `TrainExample` batch to value-loss kwargs."""
  return functools.partial(
      _critic_model_input,
      clip_range_value=algo_config.clip_range_value,
      pad_id=pad_id,
      eos_id=eos_id,
  )


# Register 'mimo' advantage estimator from advantage_shaping module.
from tunix.rl import advantage_shaping  # pylint: disable=unused-import,g-import-not-at-top
