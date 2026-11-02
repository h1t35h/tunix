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

"""Unit tests for PythonCodeEnv and code judge utilities."""

import asyncio
import email.message
import io
import json
from unittest import mock
from urllib import error as urllib_error
from urllib import request

from absl.testing import absltest
from tunix.rl.agentic.environments.code import code_env
from tunix.rl.agentic.environments.code import judge_client
from tunix.rl.agentic.environments.code import prompts
from tunix.rl.agentic.environments.code import schemas


class CodeEnvTest(absltest.TestCase):

  def test_initial_observation_and_multi_turn_recovery(self):
    env = code_env.PythonCodeEnv(
        {
            "prompt_id": "p1",
            "problem": "Write `inc(x)` returning `x + 1`.",
            "tests": ["assert inc(1) == 2", "assert inc(5) == 6"],
        },
        max_steps=3,
    )
    obs, _ = env.reset()
    self.assertIn("prompts", obs)
    self.assertEqual(obs["prompt_id"], "p1")

    # Turn 1: wrong implementation
    r1 = env.step("```python\ndef inc(x):\n    return x - 1\n```")
    self.assertFalse(r1.done)
    self.assertEqual(r1.reward, 0.0)
    self.assertIn("Passed 0/2 tests.", r1.observation)

    # Turn 2: fixed implementation -> early termination with reward 1.0
    r2 = env.step("```python\ndef inc(x):\n    return x + 1\n```")
    self.assertTrue(r2.done)
    self.assertEqual(r2.reward, 1.0)

    # TrajectoryCollectEngine calls close() before export_trajectory_extras()
    env.close()
    extras = env.export_trajectory_extras()
    self.assertTrue(extras["passed"])
    self.assertLen(extras["turn_records"], 2)

  def test_all_five_local_rule_detectors(self):
    env = code_env.PythonCodeEnv(
        {
            "prompt_id": "p2",
            "problem": "Write `f(x)`.",
            "tests": ["assert f(1) == 2"],
        },
        max_steps=4,
    )
    env.reset()

    # Turn 1: FORMAT_NO_CODE + REPEAT_LOOP
    repeated = " ".join(["one two three four five"] * 4)
    env.step(f"No code fence here: {repeated}")

    # Turn 2: SYNTAX + POST_FENCE_FAKE_OUTPUT
    env.step(
        "```python\ndef f(x)\n    return x\n```\nOutput: All tests passed!"
    )

    # Turn 3 & Turn 4: DUPLICATE_TURN
    env.step("```python\ndef f(x):\n    return x\n```")
    env.step("```python\ndef f(x):\n    # comment only\n    return x\n```")

    extras = env.export_trajectory_extras()
    all_reasons = {
        sp.reason for tr in extras["turn_records"] for sp in tr.local_flag_spans
    }
    self.assertEqual(
        all_reasons,
        {
            "FORMAT_NO_CODE",
            "REPEAT_LOOP",
            "SYNTAX",
            "POST_FENCE_FAKE_OUTPUT",
            "DUPLICATE_TURN",
        },
    )

  def test_audit_cache_key_distinct_for_different_code_ids_and_problems(self):
    code_short = "def f(x):\n    return x + 1\n"
    code_long = "def f(x):\n    res = x + 1\n    return int(res)\n"

    # Same keys {0, 1}, swapped code bodies -> distinct cache keys and rankings.
    k1 = judge_client._make_audit_cache_key(
        "Problem A", {0: code_short, 1: code_long}
    )
    k2 = judge_client._make_audit_cache_key(
        "Problem A", {0: code_long, 1: code_short}
    )
    k3 = judge_client._make_audit_cache_key(
        "Problem B", {0: code_short, 1: code_long}
    )
    self.assertNotEqual(k1, k2)
    self.assertNotEqual(k1, k3)

    mock_judge = judge_client.MockJudgeClient(latency_s=0.0)
    r1 = asyncio.run(
        mock_judge.audit_group_solutions(
            "Problem A", {0: code_short, 1: code_long}
        )
    )
    r2 = asyncio.run(
        mock_judge.audit_group_solutions(
            "Problem A", {0: code_long, 1: code_short}
        )
    )
    self.assertEqual(r1["ranking"], [0, 1])
    self.assertEqual(r2["ranking"], [1, 0])

  def test_markdown_fence_stripping_and_response_format_fallback_retry(self):
    self.assertEqual(
        judge_client._strip_code_fences(
            '```json\n{"hacks": [], "scores": {}, "ranking": [0]}\n```'
        ),
        '{"hacks": [], "scores": {}, "ranking": [0]}',
    )
    client = judge_client.OpenAICompatJudgeClient(
        base_url="http://localhost:8000/v1", max_retries=0
    )
    sent_payloads: list[dict[str, object]] = []

    class _FakeResp:

      def __init__(self, content: str) -> None:
        self._raw = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

      def read(self) -> bytes:
        return self._raw

      def __enter__(self) -> "_FakeResp":
        return self

      def __exit__(self, *args: object) -> None:
        del args

    def _fake_urlopen(req: request.Request, timeout: float = 30.0):
      del timeout
      body = json.loads(req.data.decode("utf-8"))
      sent_payloads.append(body)
      if "response_format" in body:
        raise urllib_error.HTTPError(
            req.full_url,
            400,
            "Bad Request",
            email.message.Message(),
            io.BytesIO(b"xgrammar unsupported"),
        )
      return _FakeResp(
          '```json\n{"tags": [{"id": 0, "faulty": true, "reason":'
          ' "SYNTAX"}]}\n```'
      )

    seg = schemas.Segment(
        seg_id=0,
        kind="code",
        char_span=(0, 10),
        tok_span=(0, 3),
        text_preview="def f(x)",
    )
    tr = schemas.TestResult(passed_count=0, total=1)
    with mock.patch.object(request, "urlopen", side_effect=_fake_urlopen):
      res = asyncio.run(
          client.tag_trajectory_segments("Write f(x)", [[seg]], [tr])
      )
    self.assertLen(sent_payloads, 2)
    self.assertIn("response_format", sent_payloads[0])
    self.assertNotIn("response_format", sent_payloads[1])
    self.assertEqual(
        res["segments"],
        [{"id": 0, "label": "faulty", "reason": "SYNTAX"}],
    )
    tag_prompt = prompts.build_segment_tag_prompt("Write f(x)", [[seg]], [tr])
    audit_prompt = prompts.build_group_audit_prompt(
        "Write f(x)", {0: "def f(x): return x"}
    )
    self.assertIn("no markdown code fences", tag_prompt)
    self.assertIn("no markdown code fences", audit_prompt)

  def test_visible_and_hidden_tests_zero_leakage_and_reward_gating(self):
    """Visible tests drive observation feedback while hidden tests gate reward=1.0 with zero leakage into obs."""
    env = code_env.PythonCodeEnv(
        {
            "prompt_id": "p_split",
            "problem": "Write `double(x)` returning `2 * x`.",
            "visible_tests": ["assert double(2) == 4"],
            "hidden_tests": [
                "assert double(99) == 198  # SECRET_HIDDEN_MARKER"
            ],
        },
        max_steps=3,
    )
    obs0, _ = env.reset()
    self.assertNotIn("SECRET_HIDDEN_MARKER", str(obs0["prompts"]))

    # Turn 1: solution passes visible_tests (double(2)==4) but fails hidden_tests (double(99)!=198)
    r1 = env.step("```python\ndef double(x):\n    return 4\n```")
    self.assertFalse(r1.done)
    self.assertEqual(r1.reward, 0.0)
    self.assertIn("All 1 tests passed.", r1.observation)
    self.assertNotIn("SECRET_HIDDEN_MARKER", r1.observation)
    self.assertNotIn("198", r1.observation)

    # Turn 2: genuine solution passes both visible_tests and hidden_tests
    r2 = env.step("```python\ndef double(x):\n    return 2 * x\n```")
    self.assertTrue(r2.done)
    self.assertEqual(r2.reward, 1.0)
    self.assertNotIn("SECRET_HIDDEN_MARKER", r2.observation)
    env.close()
    extras = env.export_trajectory_extras()
    self.assertTrue(extras["passed"])
    self.assertTrue(extras["turn_records"][0].test_result.all_passed)
    self.assertIsNotNone(extras["turn_records"][0].hidden_test_result)
    self.assertFalse(extras["turn_records"][0].hidden_test_result.all_passed)
    self.assertTrue(extras["turn_records"][1].hidden_test_result.all_passed)

  def test_detect_static_ast_hacks_and_audit_solution_for_hacks(self):
    """detect_static_ast_hacks and audit_solution_for_hacks catch hardcoded branches and environment tampering."""
    hardcoded = (
        "def solve(x):\n"
        "    if x == 1: return 10\n"
        "    elif x == 2: return 20\n"
        "    return 0\n"
    )
    is_hack, reason = judge_client.detect_static_ast_hacks(hardcoded)
    self.assertTrue(is_hack)
    self.assertEqual(reason, "hardcoded_cases")

    tamper = (
        "import sys\ndef solve(x):\n    sys.modules['os'] = None\n   "
        " return x\n"
    )
    is_tamper, tamper_reason = judge_client.detect_static_ast_hacks(tamper)
    self.assertTrue(is_tamper)
    self.assertEqual(tamper_reason, "environment_tampering")

    clean = "def solve(x):\n    return x * 10\n"
    is_clean_hack, _ = judge_client.audit_solution_for_hacks(
        "Multiply by 10",
        clean,
        judge_client=judge_client.MockJudgeClient(latency_s=0.0),
    )
    self.assertFalse(is_clean_hack)

    is_llm_hack, llm_reason = judge_client.audit_solution_for_hacks(
        "Multiply by 10",
        clean,
        judge_client=judge_client.MockJudgeClient(hack_ids={0}, latency_s=0.0),
    )
    self.assertTrue(is_llm_hack)
    self.assertEqual(llm_reason, "hardcoded_cases")

  def test_thinking_tags_code_extraction_and_repeat_loop_detection(self):
    """extract_last_code_fence prefers post-</think> fence and detect_local_rule_spans catches <think> loops."""
    response = (
        "<think>\nDraft inside think:\n```python\ndef f(x): return -999\n```\n"
        + " ".join(["alpha beta gamma delta epsilon zeta"] * 4)
        + "\n</think>\n```python\ndef f(x):\n    return x + 1\n```"
    )
    code, span, fence_end = code_env.extract_last_code_fence(response)
    self.assertIn("return x + 1", code)
    self.assertNotIn("-999", code)
    self.assertIsNotNone(span)

    spans, _, _ = code_env.detect_local_rule_spans(
        response, code, span, fence_end, None
    )
    reasons = {s.reason for s in spans}
    self.assertIn("REPEAT_LOOP", reasons)


if __name__ == "__main__":
  absltest.main()
