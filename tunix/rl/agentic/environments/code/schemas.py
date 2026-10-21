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

"""Data schemas for multi-turn Python coding RL environment and judge."""

import dataclasses
from typing import Any, Literal

LOCAL_RULE_REASONS: tuple[str, ...] = (
    "SYNTAX",
    "REPEAT_LOOP",
    "POST_FENCE_FAKE_OUTPUT",
    "DUPLICATE_TURN",
    "FORMAT_NO_CODE",
)

JUDGE_SEGMENT_REASONS: tuple[str, ...] = (
    "HALLUCINATED_ANALYSIS",
    "CIRCULAR_REASONING",
    "IGNORED_TRACEBACK",
    "SPECULATIVE_WORKAROUND",
    "TEST_HARDCODING",
    "SWALLOWED_EXCEPTION",
    "OK",
)


@dataclasses.dataclass(slots=True)
class TestFailure:
  """Single test failure diagnostic from sandbox execution."""

  name: str
  expected: str
  got: str
  traceback_tail: str = ""


@dataclasses.dataclass(slots=True)
class TestResult:
  """Aggregate result of running a test suite in the sandbox."""

  passed_count: int
  total: int
  failures: list[TestFailure] = dataclasses.field(default_factory=list)
  timed_out: bool = False
  error: str = ""

  @property
  def all_passed(self) -> bool:
    return (
        self.total > 0
        and self.passed_count == self.total
        and not self.timed_out
        and not self.error
    )


@dataclasses.dataclass(slots=True)
class CharSpanFlag:
  """Character span `[start, end)` flagged by a local rule or judge."""

  start: int
  end: int
  reason: str


@dataclasses.dataclass(slots=True)
class TurnRecord:
  """Per-turn record captured by `PythonCodeEnv`."""

  turn_idx: int
  response_text: str
  code: str
  code_char_span: tuple[int, int]
  test_result: TestResult
  local_flag_spans: list[CharSpanFlag] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class Segment:
  """Contiguous text/token segment within an assistant turn."""

  seg_id: int
  kind: Literal["think", "code", "post"]
  char_span: tuple[int, int]
  tok_span: tuple[int, int]
  text_preview: str
  text: str = ""


@dataclasses.dataclass(slots=True)
class SegmentTag:
  """Single segment annotation returned by the judge."""

  id: int
  label: str  # "useful" | "faulty"
  reason: str = "OK"


@dataclasses.dataclass(slots=True)
class TagTrajectoryResult:
  """Result of tagging all segments in a single trajectory."""

  segments: list[SegmentTag] = dataclasses.field(default_factory=list)
  turn_summary: list[dict[str, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class HackRecord:
  """Flagged reward hacking trajectory."""

  id: int
  reason: str


@dataclasses.dataclass(slots=True)
class GroupAuditResult:
  """Result of group solution audit."""

  hacks: list[HackRecord] = dataclasses.field(default_factory=list)
  scores: dict[str, dict[str, int]] = dataclasses.field(default_factory=dict)
  ranking: list[int] = dataclasses.field(default_factory=list)

