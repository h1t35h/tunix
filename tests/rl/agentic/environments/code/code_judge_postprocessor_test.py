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

"""Unit tests for CodeJudgeGroupPostProcessor and MockJudgeClient."""

import asyncio
from collections.abc import Sequence
from absl.testing import absltest
import numpy as np
from tunix.rl.agentic.environments.code import code_judge_postprocessor
from tunix.rl.agentic.environments.code import judge_client
from tunix.rl.agentic.environments.code import schemas


class _SimpleTokenizer:

  def encode(self, text: str) -> list[int]:
    return [ord(c) for c in text]

  def decode(
      self, token_ids: Sequence[int], skip_special_tokens: bool = False
  ) -> str:
    del skip_special_tokens
    return "".join(chr(int(c)) for c in token_ids)


class CodeJudgePostProcessorTest(absltest.TestCase):

  def test_group_postprocessor_flags_segments_detects_hacks_and_ranks(self):
    tok = _SimpleTokenizer()
    mock_judge = judge_client.MockJudgeClient(
        hack_ids={1},
        faulty_segment_reasons={0: "HALLUCINATED_ANALYSIS"},
        ranking_override=[0, 2],
    )
    proc = code_judge_postprocessor.CodeJudgeGroupPostProcessor(
        mock_judge, tok
    )

    text0 = "<think>Bad analysis</think>\n```python\ndef f(x): return x + 1\n```"
    text1 = "```python\ndef f(x):\n    if x == 1: return 2\n    return 0\n```"
    text2 = (
        "```python\ndef f(x):\n    res = x + 1\n    return int(res)\n```"
    )
    group = []
    for idx, txt in enumerate([text0, text1, text2]):
      ids = tok.encode(txt)
      group.append({
          "conversation_tokens": ids,
          "conversation_masks": [1] * len(ids),
          "assistant_turn_token_spans": [(0, len(ids))],
          "trajectory_reward": 1.0,
          "env_extras": {
              "passed": True,
              "problem": "Increment",
              "turn_records": [
                  schemas.TurnRecord(
                      turn_idx=0,
                      response_text=txt,
                      code=txt,
                      code_char_span=(0, len(txt)),
                      test_result=schemas.TestResult(passed_count=1, total=1),
                      local_flag_spans=[],
                  )
              ],
          },
      })

    out = asyncio.run(proc.process_group(group))
    # Trajectory 1 was flagged as a hack -> reward zeroed and passed=False
    self.assertFalse(out[1]["judge"]["passed"])
    self.assertTrue(out[1]["judge"]["is_hack"])
    self.assertEqual(out[1]["trajectory_reward"], 0.0)
    # Trajectory 0 and 2 remain passing with quality 1.0 and 0.5 (rho=0.5, |P|=2)
    self.assertAlmostEqual(out[0]["judge"]["quality"], 1.0)
    self.assertAlmostEqual(out[2]["judge"]["quality"], 0.5)
    # Trajectory 0 has flagged tokens from segment 0
    self.assertEqual(out[0]["conversation_token_flags"].dtype, np.int8)
    self.assertTrue(np.any(out[0]["conversation_token_flags"] == 1))

  def test_graceful_fallback_on_judge_error(self):
    tok = _SimpleTokenizer()
    mock_judge = judge_client.MockJudgeClient(fail_mode="timeout")
    proc = code_judge_postprocessor.CodeJudgeGroupPostProcessor(
        mock_judge, tok
    )
    txt = "```python\ndef f(x): return x\n```"
    ids = tok.encode(txt)
    group = [{
        "conversation_tokens": ids,
        "conversation_masks": [1] * len(ids),
        "assistant_turn_token_spans": [(0, len(ids))],
        "trajectory_reward": 1.0,
        "env_extras": {
            "passed": True,
            "problem": "Id",
            "turn_records": [
                schemas.TurnRecord(
                    turn_idx=0,
                    response_text=txt,
                    code="def f(x): return x",
                    code_char_span=(10, 28),
                    test_result=schemas.TestResult(passed_count=1, total=1),
                    local_flag_spans=[
                        schemas.CharSpanFlag(10, 20, "DUPLICATE_TURN")
                    ],
                )
            ],
        },
    }]
    out = asyncio.run(proc.process_group(group))
    self.assertTrue(out[0]["judge"]["fallback"])
    self.assertEqual(out[0]["judge"]["quality"], 1.0)
    # Local rule flags are still preserved when remote judge fails
    self.assertTrue(np.any(out[0]["conversation_token_flags"] == 1))

  def test_unwrap_traj_and_permuted_candidate_cache_isolation(self):
    from tunix.rl.agentic.agents import agent_types  # pylint: disable=g-import-not-at-top

    tok = _SimpleTokenizer()
    mock_judge = judge_client.MockJudgeClient(latency_s=0.0)
    proc = code_judge_postprocessor.CodeJudgeGroupPostProcessor(
        mock_judge, tok
    )
    short_code = "```python\ndef f(x): return x + 1\n```"
    long_code = (
        "```python\ndef f(x):\n    val = x + 1\n    return int(val)\n```"
    )

    def _make_item(txt: str) -> agent_types.TrajectoryItem:
      ids = tok.encode(txt)
      return agent_types.TrajectoryItem(
          traj={
              "conversation_tokens": ids,
              "conversation_masks": [1] * len(ids),
              "assistant_turn_token_spans": [(0, len(ids))],
              "trajectory_reward": 1.0,
              "env_extras": {
                  "passed": True,
                  "problem": "Inc",
                  "turn_records": [
                      schemas.TurnRecord(
                          turn_idx=0,
                          response_text=txt,
                          code=txt,
                          code_char_span=(0, len(txt)),
                          test_result=schemas.TestResult(
                              passed_count=1, total=1
                          ),
                          local_flag_spans=[],
                      )
                  ],
              },
          }
      )

    g1 = asyncio.run(
        proc.process_group([_make_item(short_code), _make_item(long_code)])
    )
    g2 = asyncio.run(
        proc.process_group([_make_item(long_code), _make_item(short_code)])
    )
    self.assertAlmostEqual(g1[0].traj["judge"]["quality"], 1.0)
    self.assertAlmostEqual(g1[1].traj["judge"]["quality"], 0.5)
    self.assertAlmostEqual(g2[0].traj["judge"]["quality"], 0.5)
    self.assertAlmostEqual(g2[1].traj["judge"]["quality"], 1.0)

  def test_passing_turn_code_protection_and_failed_turn_cap_and_prompt_details(
      self,
  ):
    from tunix.rl.agentic.environments.code import prompts  # pylint: disable=g-import-not-at-top

    tok = _SimpleTokenizer()
    # Suppose the judge tries to flag every segment (0..5) as faulty.
    mock_judge = judge_client.MockJudgeClient(
        faulty_segment_reasons={
            i: "SPECULATIVE_WORKAROUND" for i in range(10)
        },
        latency_s=0.0,
    )
    proc = code_judge_postprocessor.CodeJudgeGroupPostProcessor(
        mock_judge, tok, max_faulty_segments_per_turn=2
    )

    # Passing trajectory: single code segment -> LLM judge flag on code must be ignored!
    pass_txt = "```python\ndef solve(x):\n    return x + 1\n```\nHope this helps!"
    pass_ids = tok.encode(pass_txt)
    pass_item = {
        "conversation_tokens": pass_ids,
        "conversation_masks": [1] * len(pass_ids),
        "assistant_turn_token_spans": [(0, len(pass_ids))],
        "trajectory_reward": 1.0,
        "env_extras": {
            "passed": True,
            "problem": "Add one",
            "turn_records": [
                schemas.TurnRecord(
                    turn_idx=0,
                    response_text=pass_txt,
                    code="def solve(x):\n    return x + 1\n",
                    code_char_span=(10, 40),
                    test_result=schemas.TestResult(passed_count=2, total=2),
                    local_flag_spans=[],
                )
            ],
        },
    }

    # Failing trajectory with multiple segments: at most 2 segments flagged by LLM judge.
    fail_txt = (
        "<think>\nStep 1: First reasoning block with over eighty characters so it stays separate.\n\n"
        "Step 2: Second reasoning block with over eighty characters so it stays separate.\n\n"
        "Step 3: Third reasoning block with over eighty characters so it stays separate.\n</think>\n"
        "```python\ndef solve(x):\n    return x - 1\n```"
    )
    fail_ids = tok.encode(fail_txt)
    fail_tr = schemas.TestResult(
        passed_count=0,
        total=1,
        failures=[
            schemas.TestFailure(
                name="test_1",
                expected="2",
                got="0",
                traceback_tail="AssertionError: expected 2 got 0",
            )
        ],
    )
    fail_item = {
        "conversation_tokens": fail_ids,
        "conversation_masks": [1] * len(fail_ids),
        "assistant_turn_token_spans": [(0, len(fail_ids))],
        "trajectory_reward": 0.0,
        "env_extras": {
            "passed": False,
            "problem": "Add one",
            "turn_records": [
                schemas.TurnRecord(
                    turn_idx=0,
                    response_text=fail_txt,
                    code="def solve(x):\n    return x - 1\n",
                    code_char_span=(200, 230),
                    test_result=fail_tr,
                    local_flag_spans=[],
                )
            ],
        },
    }

    out = asyncio.run(proc.process_group([pass_item, fail_item]))
    # Passing trajectory code segment and brief polite closing remark are NOT flagged
    self.assertEqual(int(np.sum(out[0]["conversation_token_flags"])), 0)
    # Failing trajectory had 4 segments all marked faulty by mock judge, but capped at 2
    self.assertEqual(
        out[1]["judge"]["reason_hist"].get("SPECULATIVE_WORKAROUND", 0), 2
    )

    # Prompt includes full segment text > 120 chars and test failure diagnostics
    long_seg = schemas.Segment(
        seg_id=0,
        kind="code",
        char_span=(0, 200),
        tok_span=(0, 200),
        text_preview="x" * 120,
        text="x" * 150 + "TAIL_MARKER",
    )
    prompt_str = prompts.build_segment_tag_prompt(
        "Add one", [[long_seg]], [fail_tr]
    )
    self.assertIn("TAIL_MARKER", prompt_str)
    self.assertIn("AssertionError: expected 2 got 0", prompt_str)

  def test_forwards_max_faulty_segments_per_turn_to_judge_client(self):
    """Verifies max_faulty_segments_per_turn is forwarded from postprocessor to judge_client and prompt."""
    from unittest import mock  # pylint: disable=g-import-not-at-top
    from tunix.rl.agentic.environments.code import prompts  # pylint: disable=g-import-not-at-top

    tok = _SimpleTokenizer()
    captured_kwargs = {}

    class _SpyJudgeClient(judge_client.MockJudgeClient):

      async def tag_trajectory_segments(
          self,
          problem,
          turn_segments,
          test_results,
          *,
          max_faulty_segments_per_turn=2,
      ):
        captured_kwargs["max_faulty_segments_per_turn"] = (
            max_faulty_segments_per_turn
        )
        return await super().tag_trajectory_segments(
            problem,
            turn_segments,
            test_results,
            max_faulty_segments_per_turn=max_faulty_segments_per_turn,
        )

    spy_judge = _SpyJudgeClient(latency_s=0.0)
    proc = code_judge_postprocessor.CodeJudgeGroupPostProcessor(
        spy_judge, tok, max_faulty_segments_per_turn=4
    )
    txt = "<think>Thinking step</think>\n```python\ndef solve(x):\n    return x\n```"
    ids = tok.encode(txt)
    item = {
        "conversation_tokens": ids,
        "conversation_masks": [1] * len(ids),
        "assistant_turn_token_spans": [(0, len(ids))],
        "trajectory_reward": 0.0,
        "env_extras": {
            "passed": False,
            "problem": "Solve",
            "turn_records": [
                schemas.TurnRecord(
                    turn_idx=0,
                    response_text=txt,
                    code="def solve(x):\n    return x\n",
                    code_char_span=(29, len(txt)),
                    test_result=schemas.TestResult(passed_count=0, total=1),
                    local_flag_spans=[],
                )
            ],
        },
    }
    asyncio.run(proc.process_group([item]))
    self.assertEqual(captured_kwargs.get("max_faulty_segments_per_turn"), 4)

    # Also verify OpenAICompatJudgeClient forwards max_faulty_segments_per_turn to build_segment_tag_prompt
    client = judge_client.OpenAICompatJudgeClient(
        base_url="http://localhost:8000", model="mock-judge"
    )
    seg = schemas.Segment(
        seg_id=0,
        kind="think",
        char_span=(0, 10),
        tok_span=(0, 10),
        text_preview="think",
        text="think",
    )
    with (
        mock.patch.object(
            prompts,
            "build_segment_tag_prompt",
            wraps=prompts.build_segment_tag_prompt,
        ) as spy_prompt,
        mock.patch.object(
            client,
            "_post_json_sync",
            return_value={"segments": [], "turn_summary": []},
        ),
    ):
      asyncio.run(
          client.tag_trajectory_segments(
              "Solve",
              [[seg]],
              [schemas.TestResult(passed_count=0, total=1)],
              max_faulty_segments_per_turn=4,
          )
      )
      spy_prompt.assert_called_once()
      self.assertEqual(
          spy_prompt.call_args.kwargs.get("max_faulty_segments_per_turn"), 4
      )

  def test_multiturn_global_seg_id_no_collision(self):
    """Verifies multi-turn trajectories assign globally unique seg_ids across turns without collision."""
    tok = _SimpleTokenizer()
    t0_txt = "<think>Turn 0 reasoning step that is clean.</think>\n```python\ndef f(x):\n    return x - 1\n```"
    t1_txt = "<think>Turn 1 reasoning step that is faulty.</think>\n```python\ndef f(x):\n    return x + 1\n```"
    obs_txt = "\nObservation: test failed\n"
    full_txt = t0_txt + obs_txt + t1_txt
    ids = tok.encode(full_txt)
    t0_end = len(t0_txt)
    t1_start = len(t0_txt) + len(obs_txt)
    t1_end = len(full_txt)
    masks = [
        1 if (0 <= i < t0_end or t1_start <= i < t1_end) else 0
        for i in range(len(ids))
    ]

    seen_turn_seg_ids = []

    class _InspectingJudge(judge_client.MockJudgeClient):

      async def tag_trajectory_segments(
          self,
          problem,
          turn_segments,
          test_results,
          *,
          max_faulty_segments_per_turn=2,
      ):
        del problem, test_results, max_faulty_segments_per_turn
        for turn_segs in turn_segments:
          seen_turn_seg_ids.append([s.seg_id for s in turn_segs])
        # Flag ONLY Turn 1's first segment (which has global seg_id = turn_segments[1][0].seg_id)
        target_seg_id = turn_segments[1][0].seg_id
        return {
            "segments": [
                {
                    "id": target_seg_id,
                    "label": "faulty",
                    "reason": "IGNORED_TRACEBACK",
                }
            ],
            "turn_summary": [],
        }

    proc = code_judge_postprocessor.CodeJudgeGroupPostProcessor(
        _InspectingJudge(latency_s=0.0), tok
    )
    item = {
        "conversation_tokens": ids,
        "conversation_masks": masks,
        "assistant_turn_token_spans": [(0, t0_end), (t1_start, t1_end)],
        "trajectory_reward": 1.0,
        "env_extras": {
            "passed": True,
            "problem": "Add one",
            "turn_records": [
                schemas.TurnRecord(
                    turn_idx=0,
                    response_text=t0_txt,
                    code="def f(x):\n    return x - 1\n",
                    code_char_span=(52, len(t0_txt)),
                    test_result=schemas.TestResult(passed_count=0, total=1),
                    local_flag_spans=[],
                ),
                schemas.TurnRecord(
                    turn_idx=1,
                    response_text=t1_txt,
                    code="def f(x):\n    return x + 1\n",
                    code_char_span=(53, len(t1_txt)),
                    test_result=schemas.TestResult(passed_count=1, total=1),
                    local_flag_spans=[],
                ),
            ],
        },
    }
    out = asyncio.run(proc.process_group([item]))
    # Ensure segment IDs across Turn 0 and Turn 1 are disjoint and strictly increasing
    self.assertLen(seen_turn_seg_ids, 2)
    self.assertTrue(
        set(seen_turn_seg_ids[0]).isdisjoint(set(seen_turn_seg_ids[1]))
    )
    flags = out[0]["conversation_token_flags"]
    # Turn 0 tokens must NOT be flagged (no seg_id=0 collision!)
    self.assertEqual(int(np.sum(flags[:t0_end])), 0)
    # Turn 1's <think> segment tokens MUST be flagged
    self.assertGreater(int(np.sum(flags[t1_start:t1_end])), 0)


if __name__ == "__main__":
  absltest.main()
