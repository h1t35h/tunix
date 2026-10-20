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

"""Prompt templates and JSON schemas for Call A (segment tag) and Call B (group audit)."""

from collections.abc import Sequence
from typing import Any
from tunix.rl.agentic.environments.code import schemas

Segment = schemas.Segment
TestResult = schemas.TestResult

SEGMENT_TAG_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "label": {
                        "type": "string",
                        "enum": ["useful", "faulty"],
                    },
                    "reason": {
                        "type": "string",
                        "enum": list(schemas.JUDGE_SEGMENT_REASONS),
                    },
                },
                "required": ["id", "label", "reason"],
            },
        },
        "turn_summary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "turn": {"type": "integer"},
                    "root_cause_identified": {"type": "boolean"},
                },
                "required": ["turn", "root_cause_identified"],
            },
        },
    },
    "required": ["segments", "turn_summary"],
}

GROUP_AUDIT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hacks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "reason"],
            },
        },
        "scores": {"type": "object"},
        "ranking": {
            "type": "array",
            "items": {"type": "integer"},
        },
    },
    "required": ["hacks", "scores", "ranking"],
}


SEGMENT_TAG_SYSTEM_PROMPT: str = (
    "You are an expert Python code reviewer performing root-cause credit "
    "assignment on a multi-turn coding trajectory. Be selective: only flag "
    "segments that directly introduce or perpetuate the root-cause bug. "
    "Never flag correct boilerplate, imports, function signatures, or "
    "passing code."
)


def build_segment_tag_prompt(
    problem: str,
    segments_by_turn: Sequence[Sequence[Segment]],
    test_results: Sequence[TestResult],
    *,
    max_faulty_segments_per_turn: int = 2,
) -> str:
  """Formats Call A (per-trajectory segment tagging) prompt."""
  lines = [
      "You are an expert Python code reviewer auditing a multi-turn coding"
      " trajectory.",
      f"Problem:\n{problem}\n",
      "Label each numbered segment as 'useful' (reason 'OK') or 'faulty' with"
      f" one of {list(schemas.JUDGE_SEGMENT_REASONS)}.",
      "Rules:\n"
      "1. On PASSED turns, label all code segments as 'useful' (reason 'OK').\n"
      "2. On FAILED turns, identify ONLY the 1 to"
      f" {max_faulty_segments_per_turn} root-cause segments that directly"
      " caused the failure or ignored the previous traceback. Leave all other"
      " segments (correct helpers, imports, setup, valid reasoning) as"
      " 'useful' ('OK').",
  ]
  for turn_idx, segs in enumerate(segments_by_turn):
    tr = test_results[turn_idx] if turn_idx < len(test_results) else None
    passed = bool(tr is not None and tr.all_passed)
    passed_str = "PASSED" if passed else "FAILED"
    lines.append(f"\n--- Turn {turn_idx + 1} ({passed_str}) ---")
    if tr is not None and not passed:
      if tr.timed_out:
        lines.append("Sandbox diagnostics: Execution timed out.")
      if tr.error:
        lines.append(f"Sandbox error: {tr.error[:400]}")
      for failure in tr.failures[:2]:
        tb = (
            f" | traceback: {failure.traceback_tail[:300]}"
            if failure.traceback_tail
            else ""
        )
        lines.append(
            f"Test failure ({failure.name}): expected={failure.expected[:120]},"
            f" got={failure.got[:120]}{tb}"
        )
    for seg in segs:
      seg_content = (seg.text or seg.text_preview)[:1000]
      lines.append(
          f"[Segment {seg.seg_id} ({seg.kind})]: {seg_content!r}"
      )
  lines.append(
      "\nRequired JSON keys: "
      '{"segments": [{"id": <int>, "label": "useful"|"faulty", "reason": "<str>"}], '
      '"turn_summary": [{"turn": <int>, "root_cause_identified": <bool>}]} '
      '(or {"tags": [{"id": <int>, "faulty": <bool>, "reason": "<str>"}]}).\n'
      "Output ONLY compact single-line raw JSON starting with { and ending with }, "
      "with no markdown code fences."
  )
  return "\n".join(lines)


def build_group_audit_prompt(
    problem: str,
    passing_solutions: dict[int, str],
) -> str:
  """Formats Call B (group-level hack detection & quality ranking) prompt."""
  lines = [
      "You are an expert Python judge auditing passing candidate solutions for"
      " reward hacking and code quality.",
      f"Problem:\n{problem}\n",
      "Candidates:",
  ]
  for tid, code in sorted(passing_solutions.items()):
    lines.append(f"\n=== Candidate {tid} ===\n{code}")
  lines.append(
      "\nIdentify any reward-hacking candidates in 'hacks' and rank all clean"
      " passing candidate IDs from best to worst in 'ranking'.\n"
      "Required JSON keys: "
      '{"hacks": [{"id": <int>, "reason": "<str>"}], '
      '"scores": {"<id>": <float>}, "ranking": [<int>, ...]}.\n'
      "Output ONLY compact single-line raw JSON starting with { and ending with }, "
      "with no markdown code fences."
  )
  return "\n".join(lines)
