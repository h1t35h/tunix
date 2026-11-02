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

"""Group post-processor combining local rules and judge annotations (`CodeJudgeGroupPostProcessor`)."""

import asyncio
import collections
import dataclasses
import hashlib
import json
import re
import time
from urllib import error as urllib_error

from absl import logging
import numpy as np
from tunix.rl import advantage_shaping
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.environments.code import judge_client as judge_client_lib
from tunix.rl.agentic.environments.code import schemas
from tunix.rl.agentic.environments.code import segmenter

Segment = schemas.Segment
TurnRecord = schemas.TurnRecord
TrajectoryDict = dict[str, object]
TrajectoryLike = agent_types.TrajectoryItem | TrajectoryDict
detect_static_ast_hacks = judge_client_lib.detect_static_ast_hacks
audit_solution_for_hacks = judge_client_lib.audit_solution_for_hacks

_FAKE_POST_RE = re.compile(
    r"All tests passed|Output\s*:|Expected\s*:|>>>|assert\s+|Traceback|"
    r"AssertionError|Test\s+\d+\s+passed|\bPASSED\b|\bFAILED\b",
    re.IGNORECASE,
)


def _is_suspicious_post_segment(text: str) -> bool:
  """Returns True if a post-fence segment contains fake output markers or >200 chars."""
  stripped = text.strip()
  if not stripped:
    return False
  if len(stripped) > 200:
    return True
  return bool(_FAKE_POST_RE.search(stripped))


def _unwrap_traj(item: TrajectoryLike) -> TrajectoryDict:
  """Extracts the trajectory dictionary from a `TrajectoryItem` or raw dict."""
  if isinstance(item, agent_types.TrajectoryItem):
    return item.traj
  return item


def _cap_turn_token_flags(
    turn_flags: np.ndarray,
    num_turn_toks: int,
    max_ratio: float | None,
) -> np.ndarray:
  """Caps flagged tokens in a turn to floor(max_ratio * num_turn_toks)."""
  if max_ratio is None or num_turn_toks <= 0:
    return turn_flags
  max_allowed = int(np.floor(max_ratio * num_turn_toks))
  flagged_indices = np.flatnonzero(turn_flags != 0)
  if len(flagged_indices) <= max_allowed:
    return turn_flags
  capped = np.array(turn_flags, copy=True)
  capped[flagged_indices[max_allowed:]] = 0
  return capped


class CodeJudgeGroupPostProcessor:
  """Annotates each trajectory group with `conversation_token_flags` and `judge` metadata."""

  def __init__(
      self,
      judge_client: (
          judge_client_lib.OpenAICompatJudgeClient
          | judge_client_lib.MockJudgeClient
          | judge_client_lib.JudgeClient
          | None
      ) = None,
      tokenizer: segmenter.TokenizerProtocol | None = None,
      *,
      local_rules_only: bool = False,
      audit_only: bool = False,
      gar_rank_rho: float = 0.5,
      max_faulty_segments_per_turn: int = 1,
      max_flagged_segments_per_turn: int | None = None,
      max_flagged_token_ratio_per_turn: float | None = 0.35,
      max_flagged_token_frac: float | None = None,
  ) -> None:
    self.judge_client = judge_client
    self.tokenizer = tokenizer
    self.local_rules_only = local_rules_only
    self.audit_only = bool(audit_only)
    self.gar_rank_rho = gar_rank_rho
    self.max_faulty_segments_per_turn = (
        max_flagged_segments_per_turn
        if max_flagged_segments_per_turn is not None
        else max_faulty_segments_per_turn
    )
    self.max_flagged_token_ratio_per_turn = (
        max_flagged_token_frac
        if max_flagged_token_frac is not None
        else max_flagged_token_ratio_per_turn
    )
    self.max_flagged_token_frac = self.max_flagged_token_ratio_per_turn
    if self.max_faulty_segments_per_turn < 0:
      raise ValueError(
          "max_faulty_segments_per_turn must be >= 0, got"
          f" {self.max_faulty_segments_per_turn}."
      )
    if self.max_flagged_token_ratio_per_turn is not None and not (
        0.0 < self.max_flagged_token_ratio_per_turn <= 1.0
    ):
      raise ValueError(
          "max_flagged_token_ratio_per_turn must be in (0.0, 1.0] or None, got"
          f" {self.max_flagged_token_ratio_per_turn}."
      )

  async def process_group(
      self, group: list[TrajectoryLike]
  ) -> list[TrajectoryLike]:
    """Segments turns, queries judge, and populates token flags and judge dict."""
    t0 = time.monotonic()
    num_items = len(group)
    if num_items == 0:
      return group

    all_segments_by_item: list[list[list[Segment]]] = []
    all_offsets_fallback: list[float] = []
    passing_candidates: dict[int, str] = {}
    static_hack_set: set[int] = set()
    raw_passed: list[bool] = []

    for idx, item in enumerate(group):
      traj = _unwrap_traj(item)
      extras = (
          traj["env_extras"]
          if "env_extras" in traj and isinstance(traj["env_extras"], dict)
          else {}
      )
      turn_records: list[TurnRecord] = (
          list(extras["turn_records"])
          if "turn_records" in extras
          and isinstance(extras["turn_records"], list)
          else []
      )
      if "passed" in extras:
        passed_flag = bool(extras["passed"])
      elif "trajectory_reward" in traj and traj["trajectory_reward"] is not None:
        passed_flag = float(traj["trajectory_reward"]) > 0.0
      else:
        passed_flag = False
      raw_passed.append(passed_flag)
      if passed_flag:
        final_code = turn_records[-1].code if turn_records else ""
        passing_candidates[idx] = final_code
        is_static_hack, _ = detect_static_ast_hacks(final_code)
        if is_static_hack:
          static_hack_set.add(idx)

      conv_tokens = (
          list(traj["conversation_tokens"])
          if "conversation_tokens" in traj
          and traj["conversation_tokens"] is not None
          else []
      )
      raw_spans = (
          traj["assistant_turn_token_spans"]
          if "assistant_turn_token_spans" in traj
          else None
      )
      turn_spans = list(
          raw_spans or ([(0, len(conv_tokens))] if conv_tokens else [])
      )
      item_segs: list[list[Segment]] = []
      fallbacks = 0
      next_seg_id = 0
      for _, (s_tok, e_tok) in enumerate(turn_spans):
        turn_toks = conv_tokens[s_tok:e_tok]
        segs, used_fb = segmenter.segment_turn(turn_toks, self.tokenizer)
        if used_fb:
          fallbacks += 1
        reindexed_segs = [
            dataclasses.replace(s, seg_id=next_seg_id + i)
            for i, s in enumerate(segs)
        ]
        next_seg_id += len(reindexed_segs)
        item_segs.append(reindexed_segs)
      all_segments_by_item.append(item_segs)
      all_offsets_fallback.append(
          float(fallbacks) / float(max(len(turn_spans), 1))
      )

    fallback_used = False
    tag_outputs: list[dict[str, object]] = [
        {"segments": [], "turn_summary": []} for _ in range(num_items)
    ]
    audit_output: dict[str, object] = {
        "hacks": [],
        "scores": {},
        "ranking": [],
    }

    # Skip judge calls on all-fail groups (not passing_candidates) because
    # group-normalized scalar advantages are identically zero.
    if (
        passing_candidates
        and not self.local_rules_only
        and self.judge_client is not None
    ):
      try:
        first_traj = _unwrap_traj(group[0])
        first_extras = (
            first_traj["env_extras"]
            if "env_extras" in first_traj
            and isinstance(first_traj["env_extras"], dict)
            else {}
        )
        group_problem = (
            str(first_extras["problem"]) if "problem" in first_extras else ""
        )
        if self.audit_only:
          # Audit-only mode skips Call A (segment tagging) and only runs Call B.
          audit_output = await self.judge_client.audit_group_solutions(
              group_problem, passing_candidates
          )
        else:
          # Run all Call A tasks and Call B concurrently in a single gather.
          tag_tasks = []
          tag_fn = self.judge_client.tag_trajectory_segments
          for idx, item in enumerate(group):
            traj = _unwrap_traj(item)
            extras = (
                traj["env_extras"]
                if "env_extras" in traj and isinstance(traj["env_extras"], dict)
                else {}
            )
            turn_records = (
                list(extras["turn_records"])
                if "turn_records" in extras
                and isinstance(extras["turn_records"], list)
                else []
            )
            test_results = [tr.test_result for tr in turn_records]
            problem = str(extras["problem"]) if "problem" in extras else ""
            tag_tasks.append(
                tag_fn(
                    problem,
                    all_segments_by_item[idx],
                    test_results,
                    max_faulty_segments_per_turn=self.max_faulty_segments_per_turn,
                )
            )
          audit_coro = self.judge_client.audit_group_solutions(
              group_problem, passing_candidates
          )
          combined_results = list(await asyncio.gather(*tag_tasks, audit_coro))
          tag_outputs = combined_results[:num_items]
          audit_output = combined_results[num_items]
      except (
          urllib_error.URLError,
          TimeoutError,
          json.JSONDecodeError,
          KeyError,
          ValueError,
      ) as exc:
        logging.warning(
            "Judge execution failed (%s); using fallback annotations: %s",
            type(exc).__name__,
            exc,
        )
        fallback_used = True
        tag_outputs = [
            {"segments": [], "turn_summary": []} for _ in range(num_items)
        ]
        audit_output = {"hacks": [], "scores": {}, "ranking": []}

    raw_hacks = (
        audit_output["hacks"]
        if "hacks" in audit_output and isinstance(audit_output["hacks"], list)
        else []
    )
    hack_set = set(static_hack_set) | {
        int(h["id"])
        for h in raw_hacks
        if isinstance(h, dict) and "id" in h
    }
    effective_passed = np.array(
        [raw_passed[i] and (i not in hack_set) for i in range(num_items)],
        dtype=bool,
    )

    rank_values = np.ones(num_items, dtype=np.float64)
    if (
        not fallback_used
        and not self.local_rules_only
        and int(np.sum(effective_passed)) >= 2
    ):
      ranking_list = (
          list(audit_output["ranking"])
          if "ranking" in audit_output
          and isinstance(audit_output["ranking"], list)
          else []
      )
      code_to_first_rank: dict[str, float] = {}
      for pos, tid in enumerate(ranking_list):
        cand_id = int(tid)
        if 0 <= cand_id < num_items and effective_passed[cand_id]:
          code_hash = hashlib.sha256(
              passing_candidates[cand_id].strip().encode("utf-8")
          ).hexdigest()
          rank_val = float(pos + 1)
          if code_hash not in code_to_first_rank:
            code_to_first_rank[code_hash] = rank_val
          rank_values[cand_id] = code_to_first_rank[code_hash]
      qualities = advantage_shaping.quality_from_ranks(
          rank_values, effective_passed, rho=self.gar_rank_rho
      )
    else:
      qualities = np.ones(num_items, dtype=np.float32)

    elapsed_s = max(1e-6, time.monotonic() - t0)
    is_tagged_group = bool(passing_candidates and not self.audit_only)

    for idx, item in enumerate(group):
      traj = _unwrap_traj(item)
      conv_tokens = (
          list(traj["conversation_tokens"])
          if "conversation_tokens" in traj
          and traj["conversation_tokens"] is not None
          else []
      )
      conv_masks = np.asarray(
          traj["conversation_masks"]
          if "conversation_masks" in traj
          and traj["conversation_masks"] is not None
          else [1] * len(conv_tokens),
          dtype=np.int8,
      )
      raw_spans = (
          traj["assistant_turn_token_spans"]
          if "assistant_turn_token_spans" in traj
          else None
      )
      turn_spans = list(
          raw_spans or ([(0, len(conv_tokens))] if conv_tokens else [])
      )
      extras = (
          traj["env_extras"]
          if "env_extras" in traj and isinstance(traj["env_extras"], dict)
          else {}
      )
      turn_records = (
          list(extras["turn_records"])
          if "turn_records" in extras
          and isinstance(extras["turn_records"], list)
          else []
      )

      token_flags = np.zeros(len(conv_tokens), dtype=np.int8)
      reason_hist: collections.Counter[str] = collections.Counter()

      if not self.audit_only:
        raw_seg_tags = (
            tag_outputs[idx]["segments"]
            if "segments" in tag_outputs[idx]
            and isinstance(tag_outputs[idx]["segments"], list)
            else []
        )
        seg_labels: dict[int, tuple[str, str]] = {
            int(s["id"]): (
                str(s["label"]) if "label" in s else "useful",
                str(s["reason"]) if "reason" in s else "OK",
            )
            for s in raw_seg_tags
            if isinstance(s, dict) and "id" in s
        }

        for t_idx, (s_tok, e_tok) in enumerate(turn_spans):
          turn_toks = conv_tokens[s_tok:e_tok]
          t_offsets, _, _ = segmenter.compute_token_char_offsets(
              turn_toks, self.tokenizer
          )
          flagged_char_spans: list[tuple[int, int]] = []
          turn_passed = bool(
              t_idx < len(turn_records)
              and turn_records[t_idx].test_result.all_passed
          )

          local_post_flagged = False
          if t_idx < len(turn_records):
            for lspan in turn_records[t_idx].local_flag_spans:
              flagged_char_spans.append((lspan.start, lspan.end))
              reason_hist[lspan.reason] += 1
              if lspan.reason == "POST_FENCE_FAKE_OUTPUT":
                local_post_flagged = True

          llm_flagged_in_turn = 0
          if t_idx < len(all_segments_by_item[idx]):
            for seg in all_segments_by_item[idx][t_idx]:
              lbl, rsn = (
                  seg_labels[seg.seg_id]
                  if seg.seg_id in seg_labels
                  else ("useful", "OK")
              )
              if lbl == "faulty" or rsn != "OK":
                if turn_passed and seg.kind == "code":
                  continue
                if seg.kind == "post" and not _is_suspicious_post_segment(
                    seg.text or seg.text_preview
                ):
                  continue
                if (
                    self.max_faulty_segments_per_turn > 0
                    and llm_flagged_in_turn >= self.max_faulty_segments_per_turn
                ):
                  continue
                flagged_char_spans.append(seg.char_span)
                reason_hist[rsn] += 1
                llm_flagged_in_turn += 1
              elif (
                  seg.kind == "post"
                  and not local_post_flagged
                  and _is_suspicious_post_segment(seg.text or seg.text_preview)
              ):
                flagged_char_spans.append(seg.char_span)
                reason_hist["POST_FENCE_FAKE_OUTPUT"] += 1
                local_post_flagged = True

          turn_flags = segmenter.spans_to_token_flags(
              len(turn_toks), t_offsets, flagged_char_spans
          )
          turn_flags = _cap_turn_token_flags(
              turn_flags,
              len(turn_toks),
              self.max_flagged_token_ratio_per_turn,
          )
          token_flags[s_tok:e_tok] = np.maximum(
              token_flags[s_tok:e_tok], turn_flags
          )

      token_flags = np.where(conv_masks != 0, token_flags, 0).astype(np.int8)
      is_hack = idx in hack_set
      if is_hack:
        reason_hist["TEST_HARDCODING"] += 1
      eff_pass = bool(effective_passed[idx])
      turn1_passed = (
          bool(turn_records[0].test_result.all_passed)
          if turn_records
          else False
      )

      judge_meta: schemas.JudgeMetadataDict = {
          "passed": eff_pass,
          "turn1_passed": turn1_passed,
          "final_passed": eff_pass,
          "assistant_len": int(np.sum(conv_masks != 0)),
          "is_hack": is_hack,
          "quality": float(qualities[idx]),
          "reason_hist": dict(reason_hist),
          "latency_s": elapsed_s,
          "fallback": fallback_used,
          "offsets_fallback_frac": all_offsets_fallback[idx],
          "tagged_group": is_tagged_group,
      }
      traj["conversation_token_flags"] = token_flags
      traj["trajectory_reward"] = 1.0 if eff_pass else 0.0
      traj["judge"] = judge_meta

    return group
