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

"""Multi-turn Python coding RL environment, sandbox, segmenter, and judge."""

from tunix.rl.agentic.environments.code import code_env
from tunix.rl.agentic.environments.code import code_judge_postprocessor
from tunix.rl.agentic.environments.code import judge_client
from tunix.rl.agentic.environments.code import prompts
from tunix.rl.agentic.environments.code import sandbox
from tunix.rl.agentic.environments.code import schemas
from tunix.rl.agentic.environments.code import segmenter

PythonCodeEnv = code_env.PythonCodeEnv
CodeJudgeGroupPostProcessor = (
    code_judge_postprocessor.CodeJudgeGroupPostProcessor
)
JudgeClient = judge_client.JudgeClient
MockJudgeClient = judge_client.MockJudgeClient
OpenAICompatJudgeClient = judge_client.OpenAICompatJudgeClient
SandboxRunner = sandbox.SandboxRunner
CharSpanFlag = schemas.CharSpanFlag
Segment = schemas.Segment
TestFailure = schemas.TestFailure
TestResult = schemas.TestResult
TurnRecord = schemas.TurnRecord
compute_token_char_offsets = segmenter.compute_token_char_offsets
segment_turn = segmenter.segment_turn
spans_to_token_flags = segmenter.spans_to_token_flags

__all__ = [
    "CharSpanFlag",
    "CodeJudgeGroupPostProcessor",
    "JudgeClient",
    "MockJudgeClient",
    "OpenAICompatJudgeClient",
    "PythonCodeEnv",
    "SandboxRunner",
    "Segment",
    "TestFailure",
    "TestResult",
    "TurnRecord",
    "compute_token_char_offsets",
    "prompts",
    "segment_turn",
    "spans_to_token_flags",
]
