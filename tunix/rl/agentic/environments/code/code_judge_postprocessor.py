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
import dataclasses
import hashlib
import inspect
import re
import time
from typing import Any
import numpy as np
from tunix.rl import advantage_shaping
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.environments.code import judge_client as judge_client_lib
from tunix.rl.agentic.environments.code import schemas
from tunix.rl.agentic.environments.code import segmenter

Segment = schemas.Segment
TurnRecord = schemas.TurnRecord
TrajectoryLike = agent_types.TrajectoryItem | dict[str, Any]

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


def _unwrap_traj(item: TrajectoryLike) -> dict[str, Any]:
  """Extracts the trajectory dictionary from a `TrajectoryItem` or raw dict."""
  if isinstance(item, agent_types.TrajectoryItem):
    return item.traj
  return item


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
      tokenizer: Any = None,
      *,
      local_rules_only: bool = False,
      gar_rank_rho: float = 0.5,
      max_faulty_segments_per_turn: int = 2,
      max_flagged_segments_per_turn: int | None = None,
  ) -> None:
    self.judge_client = judge_client
    self.tokenizer = tokenizer
    self.local_rules_only = local_rules_only
    self.gar_rank_rho = gar_rank_rho
    self.max_faulty_segments_per_turn = (
        max_flagged_segments_per_turn
        if max_flagged_segments_per_turn is not None
        else max_faulty_segments_per_turn
    )

  async def process_group(
      self, group: list[TrajectoryLike]
  ) -> list[TrajectoryLike]:
    """Segments turns, queries judge (if enabled), and populates `conversation_token_flags` and `judge`."""
    t0 = time.monotonic()
    num_items = len(group)
    if num_items == 0:
      return group

    all_segments_by_item: list[list[list[Segment]]] = []
    all_offsets_fallback: list[float] = []
    passing_candidates: dict[int, str] = {}
    raw_passed: list[bool] = []

    for idx, item in enumerate(group):
      traj = _unwrap_traj(item)
      extras = traj.get("env_extras", {})
      turn_records: list[TurnRecord] = list(extras.get("turn_records", []))
      passed_flag = bool(
          extras.get("passed", traj.get("trajectory_reward", 0.0) > 0.0)
      )
      raw_passed.append(passed_flag)
      if passed_flag:
        final_code = turn_records[-1].code if turn_records else ""
        passing_candidates[idx] = final_code

      conv_tokens = list(traj.get("conversation_tokens", []))
      turn_spans = list(
          traj.get("assistant_turn_token_spans")
          or ([(0, len(conv_tokens))] if conv_tokens else [])
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
    tag_outputs: list[dict[str, Any]] = [
        {"segments": [], "turn_summary": []} for _ in range(num_items)
    ]
    audit_output: dict[str, Any] = {"hacks": [], "scores": {}, "ranking": []}

    if not self.local_rules_only and self.judge_client is not None:
      try:
        tag_tasks = []
        tag_fn = self.judge_client.tag_trajectory_segments
        tag_params = inspect.signature(tag_fn).parameters
        accepts_max_faulty = "max_faulty_segments_per_turn" in tag_params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in tag_params.values()
        )
        for idx, item in enumerate(group):
          traj = _unwrap_traj(item)
          extras = traj.get("env_extras", {})
          turn_records = list(extras.get("turn_records", []))
          test_results = [tr.test_result for tr in turn_records]
          problem = str(extras.get("problem", ""))
          if accepts_max_faulty:
            tag_tasks.append(
                tag_fn(
                    problem,
                    all_segments_by_item[idx],
                    test_results,
                    max_faulty_segments_per_turn=self.max_faulty_segments_per_turn,
                )
            )
          else:
            tag_tasks.append(
                tag_fn(problem, all_segments_by_item[idx], test_results)
            )
        tag_outputs = list(await asyncio.gather(*tag_tasks))
        if passing_candidates:
          first_traj = _unwrap_traj(group[0])
          problem = str(first_traj.get("env_extras", {}).get("problem", ""))
          audit_output = await self.judge_client.audit_group_solutions(
              problem, passing_candidates
          )
      except Exception:  # pylint: disable=broad-exception-caught
        fallback_used = True
        tag_outputs = [
            {"segments": [], "turn_summary": []} for _ in range(num_items)
        ]
        audit_output = {"hacks": [], "scores": {}, "ranking": []}

    hack_set = {
        int(h["id"])
        for h in audit_output.get("hacks", [])
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
      ranking_list = list(audit_output.get("ranking", []))
      code_to_first_rank: dict[str, float] = {}
      for pos, tid in enumerate(ranking_list):
        if 0 <= int(tid) < num_items and effective_passed[int(tid)]:
          code_hash = hashlib.sha256(
              passing_candidates.get(int(tid), "").strip().encode("utf-8")
          ).hexdigest()
          rank_val = float(pos + 1)
          if code_hash not in code_to_first_rank:
            code_to_first_rank[code_hash] = rank_val
          rank_values[int(tid)] = code_to_first_rank[code_hash]
      qualities = advantage_shaping.quality_from_ranks(
          rank_values, effective_passed, rho=self.gar_rank_rho
      )
    else:
      qualities = np.ones(num_items, dtype=np.float32)

    elapsed_s = max(1e-6, time.monotonic() - t0)

    for idx, item in enumerate(group):
      traj = _unwrap_traj(item)
      conv_tokens = list(traj.get("conversation_tokens", []))
      conv_masks = np.asarray(
          traj.get("conversation_masks", [1] * len(conv_tokens)), dtype=np.int8
      )
      turn_spans = list(
          traj.get("assistant_turn_token_spans")
          or ([(0, len(conv_tokens))] if conv_tokens else [])
      )
      extras = traj.get("env_extras", {})
      turn_records = list(extras.get("turn_records", []))

      token_flags = np.zeros(len(conv_tokens), dtype=np.int8)
      reason_hist: dict[str, int] = {}

      seg_labels = {
          int(s["id"]): (
              str(s.get("label", "useful")),
              str(s.get("reason", "OK")),
          )
          for s in tag_outputs[idx].get("segments", [])
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
            reason_hist[lspan.reason] = reason_hist.get(lspan.reason, 0) + 1
            if lspan.reason == "POST_FENCE_FAKE_OUTPUT":
              local_post_flagged = True

        llm_flagged_in_turn = 0
        if t_idx < len(all_segments_by_item[idx]):
          for seg in all_segments_by_item[idx][t_idx]:
            lbl, rsn = seg_labels.get(seg.seg_id, ("useful", "OK"))
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
              reason_hist[rsn] = reason_hist.get(rsn, 0) + 1
              llm_flagged_in_turn += 1
            elif (
                seg.kind == "post"
                and not local_post_flagged
                and _is_suspicious_post_segment(seg.text or seg.text_preview)
            ):
              flagged_char_spans.append(seg.char_span)
              reason_hist["POST_FENCE_FAKE_OUTPUT"] = (
                  reason_hist.get("POST_FENCE_FAKE_OUTPUT", 0) + 1
              )
              local_post_flagged = True

        turn_flags = segmenter.spans_to_token_flags(
            len(turn_toks), t_offsets, flagged_char_spans
        )
        token_flags[s_tok:e_tok] = np.maximum(
            token_flags[s_tok:e_tok], turn_flags
        )

      token_flags = np.where(conv_masks != 0, token_flags, 0).astype(np.int8)
      is_hack = idx in hack_set
      if is_hack:
        reason_hist["TEST_HARDCODING"] = (
            reason_hist.get("TEST_HARDCODING", 0) + 1
        )
      eff_pass = bool(effective_passed[idx])

      traj["conversation_token_flags"] = token_flags
      traj["trajectory_reward"] = 1.0 if eff_pass else 0.0
      traj["judge"] = {
          "passed": eff_pass,
          "is_hack": is_hack,
          "quality": float(qualities[idx]),
          "assistant_len": int(np.sum(conv_masks != 0)),
          "reason_hist": reason_hist,
          "latency_s": elapsed_s,
          "fallback": fallback_used,
          "offsets_fallback_frac": all_offsets_fallback[idx],
      }

    return group
