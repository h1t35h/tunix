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
import collections
from collections.abc import Mapping, Sequence
import re
from typing import NamedTuple

from tunix.rl.agentic.environments import base_environment
from tunix.rl.agentic.environments.code import sandbox as sandbox_lib
from tunix.rl.agentic.environments.code import schemas

CharSpanFlag = schemas.CharSpanFlag
TestResult = schemas.TestResult
TurnRecord = schemas.TurnRecord
TestSpec = Sequence[str] | Mapping[str, Sequence[str] | str]


class StepOutcome(NamedTuple):
  """Step return supporting both 4-tuple unpacking and attribute access."""

  observation: str
  reward: float
  done: bool
  info: dict[str, bool | int]


def extract_last_code_fence(
    text: str,
) -> tuple[str | None, tuple[int, int], int]:
  """Extracts the last ```python ... ``` or ``` ... ``` block from text."""
  close_think_idx = text.rfind("</think>")
  if close_think_idx != -1:
    search_start = close_think_idx + len("</think>")
    after_matches = list(
        re.finditer(
            r"```(?:python)?\s*\n(.*?)```",
            text[search_start:],
            flags=re.DOTALL,
        )
    )
    if after_matches:
      last = after_matches[-1]
      return (
          last.group(1),
          (search_start + last.start(1), search_start + last.end(1)),
          search_start + last.end(0),
      )
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
  think_match = re.search(
      r"<think>(.*?)(?:</think>|(?=```))", response_text, flags=re.DOTALL
  )
  repeat_search_start = (
      think_match.end(0) if (think_match and code is not None) else 0
  )
  token_matches = list(re.finditer(r"\S+", response_text[repeat_search_start:]))
  if len(token_matches) >= 15:
    words = [m.group(0) for m in token_matches]
    ngram_positions: dict[tuple[str, ...], list[int]] = collections.defaultdict(
        list
    )
    for idx in range(len(words) - 4):
      ng = tuple(words[idx : idx + 5])
      ngram_positions[ng].append(idx)
    for _, pos_list in ngram_positions.items():
      if len(pos_list) >= 3:
        start_char = repeat_search_start + token_matches[pos_list[0]].start()
        end_char = (
            repeat_search_start
            + token_matches[min(len(token_matches) - 1, pos_list[-1] + 4)].end()
        )
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
      single_example: Mapping[str, object] | None = None,
      *,
      group_id: int = 0,
      pair_index: int = 0,
      sandbox: sandbox_lib.SandboxRunner | None = None,
      max_steps: int = 3,
      max_obs_chars: int = 1500,
      test_timeout_s: float = 3.0,
      enable_thinking: bool = False,
      **kwargs: object,
  ) -> None:
    raw_task = (
        single_example
        if single_example is not None
        else (kwargs.pop("task") if "task" in kwargs else None)
    )
    task_dict = dict(raw_task) if isinstance(raw_task, Mapping) else {}
    super().__init__(task=task_dict, max_steps=max_steps, **kwargs)
    self.group_id = group_id
    self.pair_index = pair_index
    self.sandbox = sandbox or sandbox_lib.SandboxRunner(
        default_timeout_s=test_timeout_s
    )
    self.max_obs_chars = max_obs_chars
    self.test_timeout_s = test_timeout_s
    self.enable_thinking = bool(enable_thinking)
    self._turn_records: list[TurnRecord] = []
    self._passed = False
    self._prev_ast_dump: str | None = None
    self._closed = False

  def _problem_text(self) -> str:
    if "problem" in self.task and self.task["problem"]:
      return str(self.task["problem"])
    if "question" in self.task and self.task["question"]:
      return str(self.task["question"])
    return ""

  def _visible_tests(self) -> TestSpec:
    if "visible_tests" in self.task:
      return self.task["visible_tests"]
    if "tests" in self.task:
      return self.task["tests"]
    return []

  def _hidden_tests(self) -> TestSpec:
    if "hidden_tests" in self.task:
      return self.task["hidden_tests"]
    if "tests" in self.task:
      return self.task["tests"]
    return []

  def _format(self) -> str:
    if "format" in self.task and self.task["format"]:
      return str(self.task["format"])
    return "asserts"

  def _prompt_id(self) -> str:
    if "prompt_id" in self.task and self.task["prompt_id"]:
      return str(self.task["prompt_id"])
    return "task_0"

  def _initial_observation(self) -> dict[str, object]:
    problem = self._problem_text()
    fmt = self._format()
    format_instructions = (
        str(self.task["format_instructions"])
        if "format_instructions" in self.task
        and self.task["format_instructions"]
        else (
            "Enclose your final Python solution inside a ```python ... ``` code"
            " block."
        )
    )
    prompts = (
        str(self.task["prompts"])
        if "prompts" in self.task and self.task["prompts"]
        else f"{problem}\n\n{format_instructions}".strip()
    )
    visible_tests = self._visible_tests()
    return {
        "prompts": prompts,
        "problem": problem,
        "format_instructions": format_instructions,
        "prompt_id": self._prompt_id(),
        "tests": visible_tests,
        "format": fmt,
    }

  def reset(self) -> tuple[dict[str, object], dict[str, object]]:
    self.step_count = 0
    self._turn_records = []
    self._passed = False
    self._prev_ast_dump = None
    self._closed = False
    return self._initial_observation(), {}

  def step(self, action: object) -> StepOutcome:
    self.step_count += 1
    result = self._step_impl(action)
    done = result.done or (self.step_count >= self.max_steps)
    return StepOutcome(
        observation=str(result.observation),
        reward=result.reward,
        done=done,
        info=result.info,
    )

  def _record_turn_and_result(
      self,
      *,
      turn_idx: int,
      action_text: str,
      code: str,
      code_span: tuple[int, int],
      test_res: TestResult,
      local_spans: list[CharSpanFlag],
      observation: str,
      reward: float,
      done: bool,
      passed: bool,
      hidden_test_res: TestResult | None = None,
      visible_passed: bool | None = None,
      hidden_passed: bool | None = None,
  ) -> base_environment.EnvStepResult:
    self._turn_records.append(
        TurnRecord(
            turn_idx=turn_idx,
            response_text=action_text,
            code=code,
            code_char_span=code_span,
            test_result=test_res,
            local_flag_spans=local_spans,
            hidden_test_result=hidden_test_res,
        )
    )
    info: dict[str, bool | int] = {"passed": passed, "turn_idx": turn_idx}
    if visible_passed is not None:
      info["visible_passed"] = visible_passed
    if hidden_passed is not None:
      info["hidden_passed"] = hidden_passed
    return base_environment.EnvStepResult(
        observation=observation[: self.max_obs_chars],
        reward=reward,
        done=done,
        info=info,
    )

  def _step_impl(self, action: object) -> base_environment.EnvStepResult:
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
      return self._record_turn_and_result(
          turn_idx=turn_idx,
          action_text=action_text,
          code="",
          code_span=code_span,
          test_res=test_res,
          local_spans=local_spans,
          observation=obs,
          reward=0.0,
          done=done,
          passed=False,
      )

    if syntax_err is not None:
      err_msg = f"SyntaxError: {syntax_err.msg} (line {syntax_err.lineno})"
      test_res = TestResult(passed_count=0, total=1, error=err_msg)
      obs = f"Your code failed to parse:\n{err_msg}"
      done = self.step_count >= self.max_steps
      return self._record_turn_and_result(
          turn_idx=turn_idx,
          action_text=action_text,
          code=code,
          code_span=code_span,
          test_res=test_res,
          local_spans=local_spans,
          observation=obs,
          reward=0.0,
          done=done,
          passed=False,
      )

    visible_tests = self._visible_tests()
    hidden_tests = self._hidden_tests()
    fmt = self._format()

    def _to_run_spec(spec: TestSpec) -> TestSpec:
      if fmt == "stdio" and isinstance(spec, dict):
        return {**spec, "format": "stdio"}
      return spec

    timeout_s = (
        float(self.task["time_limit_s"])
        if "time_limit_s" in self.task and self.task["time_limit_s"] is not None
        else self.test_timeout_s
    )
    entry_point = (
        str(self.task["entry_point"])
        if "entry_point" in self.task and self.task["entry_point"]
        else None
    )
    visible_res = self.sandbox.run(
        code,
        _to_run_spec(visible_tests),
        timeout=timeout_s,
        entry_point=entry_point,
    )
    has_separate_hidden = "hidden_tests" in self.task and (
        hidden_tests != visible_tests
    )
    if visible_res.all_passed and has_separate_hidden:
      hidden_res = self.sandbox.run(
          code,
          _to_run_spec(hidden_tests),
          timeout=timeout_s,
          entry_point=entry_point,
      )
    else:
      hidden_res = visible_res

    visible_passed = visible_res.all_passed
    hidden_passed = hidden_res.all_passed
    all_passed = visible_passed and hidden_passed
    self._passed = all_passed
    done = all_passed or (self.step_count >= self.max_steps)
    reward = 1.0 if all_passed else 0.0

    if visible_passed:
      obs = f"All {visible_res.total} tests passed."
    else:
      fail_summaries = [
          f"Passed {visible_res.passed_count}/{visible_res.total} tests."
      ]
      if visible_res.timed_out:
        fail_summaries.append("Execution timed out.")
      for f in visible_res.failures[:3]:
        detail = f"{f.name}: expected {f.expected}, got {f.got}"
        if f.traceback_tail:
          detail += f"\n{f.traceback_tail}"
        fail_summaries.append(detail)
      obs = "\n".join(fail_summaries)

    return self._record_turn_and_result(
        turn_idx=turn_idx,
        action_text=action_text,
        code=code,
        code_span=code_span,
        test_res=visible_res,
        local_spans=local_spans,
        observation=obs,
        reward=reward,
        done=done,
        passed=all_passed,
        hidden_test_res=hidden_res if has_separate_hidden else None,
        visible_passed=visible_passed,
        hidden_passed=hidden_passed,
    )

  def export_trajectory_extras(self) -> dict[str, object]:
    return {
        "turn_records": list(self._turn_records),
        "passed": bool(self._passed),
        "format": self._format(),
        "prompt_id": self._prompt_id(),
        "problem": self._problem_text(),
        "tests": self._visible_tests(),
    }

  def close(self) -> None:
    # Must NOT clear _turn_records because TrajectoryCollectEngine calls close()
    # before export_trajectory_extras() in Token mode.
    self._closed = True
