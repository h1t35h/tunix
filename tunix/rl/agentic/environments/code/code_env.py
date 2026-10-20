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

"""Multi-turn Python coding RL environment (`PythonCodeEnv`)."""

import ast
import re
from typing import Any, NamedTuple
from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.environments.code import sandbox as sandbox_lib
from tunix.rl.agentic.environments.code import schemas

CharSpanFlag = schemas.CharSpanFlag
TestResult = schemas.TestResult
TurnRecord = schemas.TurnRecord


class StepOutcome(NamedTuple):
  """Step return supporting both 4-tuple unpacking and attribute access."""

  observation: Any
  reward: float
  done: bool
  info: dict[str, Any]


def extract_last_code_fence(
    text: str,
) -> tuple[str | None, tuple[int, int], int]:
  """Extracts the last ```python ... ``` or ``` ... ``` block from text."""
  matches = list(
      re.finditer(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
  )
  if not matches:
    return None, (0, len(text)), len(text)
  last = matches[-1]
  return last.group(1), last.span(1), last.end(0)


def detect_local_rule_spans(
    response_text: str,
    code: str | None,
    code_span: tuple[int, int],
    fence_end: int,
    prev_ast_dump: str | None,
) -> tuple[list[CharSpanFlag], str | None, SyntaxError | None]:
  """Applies all 5 deterministic local rule detectors."""
  spans: list[CharSpanFlag] = []

  # 1. REPEAT_LOOP: any whitespace 5-gram repeated >= 3x
  token_matches = list(re.finditer(r"\S+", response_text))
  if len(token_matches) >= 15:
    words = [m.group(0) for m in token_matches]
    ngram_positions: dict[tuple[str, ...], list[int]] = {}
    for idx in range(len(words) - 4):
      ng = tuple(words[idx : idx + 5])
      ngram_positions.setdefault(ng, []).append(idx)
    for _, pos_list in ngram_positions.items():
      if len(pos_list) >= 3:
        start_char = token_matches[pos_list[0]].start()
        end_char = token_matches[
            min(len(token_matches) - 1, pos_list[-1] + 4)
        ].end()
        spans.append(CharSpanFlag(start_char, end_char, "REPEAT_LOOP"))
        break

  # 2. FORMAT_NO_CODE
  if code is None:
    spans.append(CharSpanFlag(0, len(response_text), "FORMAT_NO_CODE"))
    return spans, None, None

  # 3. POST_FENCE_FAKE_OUTPUT
  post_text = response_text[fence_end:]
  if post_text.strip():
    fake_patterns = (
        r"All tests passed",
        r"Output\s*:",
        r">>>",
        r"AssertionError",
        r"Test\s+\d+\s+passed",
        r"PASSED",
    )
    if any(re.search(pat, post_text, re.IGNORECASE) for pat in fake_patterns):
      spans.append(
          CharSpanFlag(fence_end, len(response_text), "POST_FENCE_FAKE_OUTPUT")
      )

  # 4. SYNTAX & 5. DUPLICATE_TURN
  try:
    parsed = ast.parse(code)
    curr_ast_dump = ast.dump(
        parsed, annotate_fields=True, include_attributes=False
    )
  except SyntaxError as exc:
    spans.append(CharSpanFlag(code_span[0], code_span[1], "SYNTAX"))
    return spans, None, exc

  if prev_ast_dump is not None and curr_ast_dump == prev_ast_dump:
    spans.append(CharSpanFlag(code_span[0], code_span[1], "DUPLICATE_TURN"))

  return spans, curr_ast_dump, None


class PythonCodeEnv(base_environment.BaseTaskEnv):
  """Multi-turn Python coding environment with sandbox test feedback."""

  def __init__(
      self,
      single_example: dict[str, Any] | None = None,
      *,
      group_id: int = 0,
      pair_index: int = 0,
      sandbox: sandbox_lib.SandboxRunner | None = None,
      max_steps: int = 3,
      max_obs_chars: int = 1500,
      test_timeout_s: float = 3.0,
      **kwargs: Any,
  ) -> None:
    task_dict = dict(single_example or kwargs.pop("task", None) or {})
    super().__init__(task=task_dict, max_steps=max_steps, **kwargs)
    self.group_id = group_id
    self.pair_index = pair_index
    self.sandbox = sandbox or sandbox_lib.SandboxRunner(
        default_timeout_s=test_timeout_s
    )
    self.max_obs_chars = max_obs_chars
    self.test_timeout_s = test_timeout_s
    self._turn_records: list[TurnRecord] = []
    self._passed = False
    self._prev_ast_dump: str | None = None
    self._closed = False

  def _initial_observation(self) -> dict[str, Any]:
    problem = str(self.task.get("problem") or self.task.get("question") or "")
    fmt = str(self.task.get("format") or "asserts")
    format_instructions = str(
        self.task.get("format_instructions")
        or (
            "Enclose your final Python solution inside a ```python ... ``` code"
            " block."
        )
    )
    prompts = str(
        self.task.get("prompts") or f"{problem}\n\n{format_instructions}".strip()
    )
    return {
        "prompts": prompts,
        "problem": problem,
        "format_instructions": format_instructions,
        "prompt_id": str(self.task.get("prompt_id", "task_0")),
        "tests": self.task.get("tests", []),
        "format": fmt,
    }

  def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
    self.step_count = 0
    self._turn_records = []
    self._passed = False
    self._prev_ast_dump = None
    self._closed = False
    return self._initial_observation(), {}

  def step(self, action: Any) -> StepOutcome:
    self.step_count += 1
    result = self._step_impl(action)
    done = result.done or (self.step_count >= self.max_steps)
    return StepOutcome(
        observation=result.observation,
        reward=result.reward,
        done=done,
        info=result.info,
    )

  def _step_impl(self, action: Any) -> base_environment.EnvStepResult:
    action_text = str(action)
    turn_idx = max(0, self.step_count - 1)

    code, code_span, fence_end = extract_last_code_fence(action_text)
    local_spans, ast_dump, syntax_err = detect_local_rule_spans(
        action_text, code, code_span, fence_end, self._prev_ast_dump
    )
    if ast_dump is not None:
      self._prev_ast_dump = ast_dump

    if code is None:
      test_res = TestResult(
          passed_count=0, total=1, error="No code block found"
      )
      obs = (
          "No code block found. Please provide your solution inside a"
          " ```python ... ``` block."
      )
      done = self.step_count >= self.max_steps
      self._turn_records.append(
          TurnRecord(
              turn_idx=turn_idx,
              response_text=action_text,
              code="",
              code_char_span=code_span,
              test_result=test_res,
              local_flag_spans=local_spans,
          )
      )
      return base_environment.EnvStepResult(
          observation=obs[: self.max_obs_chars],
          reward=0.0,
          done=done,
          info={"passed": False, "turn_idx": turn_idx},
      )

    if syntax_err is not None:
      err_msg = f"SyntaxError: {syntax_err.msg} (line {syntax_err.lineno})"
      test_res = TestResult(passed_count=0, total=1, error=err_msg)
      obs = f"Your code failed to parse:\n{err_msg}"
      done = self.step_count >= self.max_steps
      self._turn_records.append(
          TurnRecord(
              turn_idx=turn_idx,
              response_text=action_text,
              code=code,
              code_char_span=code_span,
              test_result=test_res,
              local_flag_spans=local_spans,
          )
      )
      return base_environment.EnvStepResult(
          observation=obs[: self.max_obs_chars],
          reward=0.0,
          done=done,
          info={"passed": False, "turn_idx": turn_idx},
      )

    tests_spec = self.task.get("tests", [])
    fmt = self.task.get("format", "asserts")
    if fmt == "stdio" and isinstance(tests_spec, dict):
      run_spec: Any = {**tests_spec, "format": "stdio"}
    else:
      run_spec = tests_spec

    test_res = self.sandbox.run(
        code,
        run_spec,
        timeout=self.task.get("time_limit_s", self.test_timeout_s),
        entry_point=self.task.get("entry_point"),
    )
    all_passed = test_res.all_passed
    self._passed = all_passed
    done = all_passed or (self.step_count >= self.max_steps)
    reward = 1.0 if all_passed else 0.0

    if all_passed:
      obs = f"All {test_res.total} tests passed."
    else:
      fail_summaries = [
          f"Passed {test_res.passed_count}/{test_res.total} tests."
      ]
      if test_res.timed_out:
        fail_summaries.append("Execution timed out.")
      for f in test_res.failures[:3]:
        detail = f"{f.name}: expected {f.expected}, got {f.got}"
        if f.traceback_tail:
          detail += f"\n{f.traceback_tail}"
        fail_summaries.append(detail)
      obs = "\n".join(fail_summaries)[: self.max_obs_chars]

    self._turn_records.append(
        TurnRecord(
            turn_idx=turn_idx,
            response_text=action_text,
            code=code,
            code_char_span=code_span,
            test_result=test_res,
            local_flag_spans=local_spans,
        )
    )
    return base_environment.EnvStepResult(
        observation=obs,
        reward=reward,
        done=done,
        info={"passed": all_passed, "turn_idx": turn_idx},
    )

  def export_trajectory_extras(self) -> dict[str, Any]:
    return {
        "turn_records": list(self._turn_records),
        "passed": bool(self._passed),
        "format": str(self.task.get("format", "asserts")),
        "prompt_id": str(self.task.get("prompt_id", "task_0")),
        "problem": str(
            self.task.get("problem") or self.task.get("question") or ""
        ),
        "tests": self.task.get("tests", []),
    }

  def close(self) -> None:
    # Must NOT clear _turn_records because TrajectoryCollectEngine calls close()
    # before export_trajectory_extras() in Token mode.
    self._closed = True
