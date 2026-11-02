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

"""Judge clients for trajectory segment tagging and group solution auditing."""

import ast
import asyncio
from collections.abc import Sequence
import concurrent.futures
import copy
import hashlib
import json
import re
from typing import Literal, Protocol, runtime_checkable
from urllib import error as urllib_error
from urllib import request

from tunix.rl.agentic.environments.code import prompts as prompts_lib
from tunix.rl.agentic.environments.code import schemas

Segment = schemas.Segment
TestResult = schemas.TestResult

_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL | re.IGNORECASE
)
_HARDCODED_IF_ELIF_RE = re.compile(
    r"if\s+.*==.*return\s+.*\n\s*elif\s+.*==.*return", re.MULTILINE
)
_TAMPER_ATTRS = frozenset({"modules", "_exit", "__builtins__", "builtins"})
_TAMPER_MODULES = frozenset({"builtins", "unittest.mock", "mock"})


def _is_literal_return_branch(stmt_list: list[ast.stmt]) -> bool:
  """Returns True if a single-statement branch returns a literal constant."""
  if len(stmt_list) != 1 or not isinstance(stmt_list[0], ast.Return):
    return False
  val = stmt_list[0].value
  if isinstance(val, ast.Constant):
    return True
  if isinstance(val, ast.UnaryOp) and isinstance(val.operand, ast.Constant):
    return True
  if isinstance(val, (ast.List, ast.Tuple, ast.Set)):
    return all(isinstance(elt, ast.Constant) for elt in val.elts)
  return False


def _count_eq_literal_branches(node: ast.If) -> int:
  """Counts chained if/elif equality comparisons returning constants."""
  count = 0
  curr: ast.If | None = node
  while curr is not None:
    has_eq = any(
        isinstance(op, (ast.Eq, ast.In))
        for sub in ast.walk(curr.test)
        if isinstance(sub, ast.Compare)
        for op in sub.ops
    )
    if has_eq and _is_literal_return_branch(curr.body):
      count += 1
    if len(curr.orelse) == 1 and isinstance(curr.orelse[0], ast.If):
      curr = curr.orelse[0]
    else:
      break
  return count


def detect_static_ast_hacks(code: str) -> tuple[bool, str]:
  """Detects hardcoded input-dispatch branches or test tampering via AST/regex."""
  if _HARDCODED_IF_ELIF_RE.search(code):
    return True, "hardcoded_cases"
  try:
    tree = ast.parse(code)
  except SyntaxError:
    return False, "OK"

  for node in ast.walk(tree):
    if isinstance(node, ast.Import):
      for alias in node.names:
        if alias.name in _TAMPER_MODULES:
          return True, "environment_tampering"
    elif isinstance(node, ast.ImportFrom):
      if (node.module or "") in _TAMPER_MODULES:
        return True, "environment_tampering"
    elif isinstance(node, ast.Attribute):
      if node.attr in _TAMPER_ATTRS and isinstance(node.value, ast.Name):
        if node.value.id in ("sys", "os", "builtins", "__builtins__"):
          return True, "environment_tampering"
    elif isinstance(node, ast.Name) and node.id == "__builtins__":
      return True, "environment_tampering"
    elif isinstance(node, ast.If):
      if _count_eq_literal_branches(node) >= 2:
        return True, "hardcoded_cases"
  return False, "OK"


def audit_solution_for_hacks(
    problem: str,
    code: str,
    judge_client: "JudgeClient | None" = None,
) -> tuple[bool, str]:
  """Runs static AST hack detection and optional LLM judge audit on code."""
  is_static_hack, reason = detect_static_ast_hacks(code)
  if is_static_hack:
    return True, reason
  if judge_client is None:
    return False, "OK"

  async def _run_audit() -> dict[str, object]:
    return await judge_client.audit_group_solutions(problem, {0: code})

  try:
    loop = asyncio.get_running_loop()
  except RuntimeError:
    loop = None

  if loop is not None and loop.is_running():
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
      res = pool.submit(lambda: asyncio.run(_run_audit())).result()
  else:
    res = asyncio.run(_run_audit())

  hacks = res["hacks"] if "hacks" in res and isinstance(res["hacks"], list) else []
  for h in hacks:
    if isinstance(h, dict) and "id" in h and int(h["id"]) == 0:
      hack_reason = str(h["reason"]) if "reason" in h else "llm_audit_hack"
      return True, hack_reason
  return False, "OK"


def _strip_code_fences(text: str) -> str:
  """Strips surrounding markdown code fences if present."""
  stripped = text.strip()
  match = _CODE_FENCE_RE.match(stripped)
  if match:
    return match.group(1).strip()
  return stripped


def _make_audit_cache_key(
    problem: str, passing_solutions: dict[int, str]
) -> str:
  """Builds a deterministic cache key for problem and passing solutions."""
  key_obj = (
      problem,
      tuple(
          sorted(
              (
                  int(tid),
                  hashlib.sha256(code.strip().encode("utf-8")).hexdigest(),
              )
              for tid, code in passing_solutions.items()
          )
      ),
  )
  return hashlib.sha256(json.dumps(key_obj).encode("utf-8")).hexdigest()


class AuditCache:
  """Unified deterministic in-memory cache for group solution audits."""

  def __init__(self) -> None:
    self._store: dict[str, dict[str, object]] = {}

  def get(
      self, problem: str, passing_solutions: dict[int, str]
  ) -> dict[str, object] | None:
    key = _make_audit_cache_key(problem, passing_solutions)
    if key in self._store:
      return copy.deepcopy(self._store[key])
    return None

  def set(
      self,
      problem: str,
      passing_solutions: dict[int, str],
      result: dict[str, object],
  ) -> None:
    key = _make_audit_cache_key(problem, passing_solutions)
    self._store[key] = copy.deepcopy(result)

  def __contains__(self, key: str) -> bool:
    return key in self._store

  def __getitem__(self, key: str) -> dict[str, object]:
    return self._store[key]

  def __setitem__(self, key: str, value: dict[str, object]) -> None:
    self._store[key] = value

  def __len__(self) -> int:
    return len(self._store)

  def clear(self) -> None:
    self._store.clear()


@runtime_checkable
class JudgeClient(Protocol):
  """Protocol for trajectory segment tagging and group solution auditing."""

  async def tag_trajectory_segments(
      self,
      problem: str,
      segments_by_turn: Sequence[Sequence[Segment]],
      test_results: Sequence[TestResult],
      *,
      max_faulty_segments_per_turn: int = 1,
  ) -> dict[str, object]:
    """Tags trajectory segments as useful or faulty."""
    ...

  async def audit_group_solutions(
      self,
      problem: str,
      passing_solutions: dict[int, str],
  ) -> dict[str, object]:
    """Audits passing solutions for reward hacking and ranks clean solutions."""
    ...


class OpenAICompatJudgeClient:
  """Async OpenAI-compatible HTTP client for vLLM judge endpoints."""

  def __init__(
      self,
      base_url: str = "http://localhost:8000/v1",
      model: str = "Qwen/Qwen3-Coder-Next",
      *,
      api_key: str = "EMPTY",
      max_concurrency: int = 64,
      timeout_s: float = 30.0,
      max_retries: int = 2,
      max_tokens: int = 512,
      enable_thinking: bool = False,
  ) -> None:
    self.base_url = base_url.rstrip("/")
    self.model = model
    self.api_key = api_key
    self.max_concurrency = max_concurrency
    self.timeout_s = timeout_s
    self.max_retries = max_retries
    self.max_tokens = max_tokens
    self.enable_thinking = enable_thinking
    self._sem = asyncio.Semaphore(max_concurrency)
    self._audit_cache = AuditCache()

  def _post_json_sync(self, payload: dict[str, object]) -> dict[str, object]:
    req_payload: dict[str, object] = {
        "max_tokens": self.max_tokens,
        "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        **payload,
    }
    data = json.dumps(req_payload).encode("utf-8")
    req = request.Request(
        f"{self.base_url}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=self.timeout_s) as resp:
      body = json.loads(resp.read().decode("utf-8"))
    content = str(body["choices"][0]["message"]["content"])
    cleaned = _strip_code_fences(content)
    try:
      return json.loads(cleaned)
    except json.JSONDecodeError:
      if '"segments"' in cleaned or '"hacks"' in cleaned or '"tags"' in cleaned:
        seen_ids: set[int] = set()
        segs: list[schemas.SegmentTagDict] = []
        for m in re.finditer(
            r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"label"\s*:\s*"(useful|faulty)"\s*,\s*"reason"\s*:\s*"([^"]+)"\s*\}',
            cleaned,
        ):
          sid = int(m.group(1))
          if sid not in seen_ids:
            seen_ids.add(sid)
            segs.append(
                {"id": sid, "label": m.group(2), "reason": m.group(3)}
            )
        return {
            "segments": segs,
            "turn_summary": [],
            "hacks": [],
            "scores": {},
            "ranking": [],
        }
      raise

  async def _chat_json(
      self,
      prompt: str,
      schema: dict[str, object],
      *,
      system_prompt: str | None = None,
  ) -> dict[str, object]:
    messages: list[dict[str, str]] = []
    if system_prompt:
      messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, object] = {
        "model": self.model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": self.max_tokens,
        "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judge_response", "schema": schema},
        },
    }
    last_exc: Exception | None = None
    for attempt in range(self.max_retries + 1):
      try:
        async with self._sem:
          loop = asyncio.get_running_loop()
          try:
            return await loop.run_in_executor(
                None, self._post_json_sync, payload
            )
          except urllib_error.HTTPError as http_exc:
            if http_exc.code in (400, 500) and "response_format" in payload:
              fallback_payload = {
                  k: v for k, v in payload.items() if k != "response_format"
              }
              return await loop.run_in_executor(
                  None, self._post_json_sync, fallback_payload
              )
            raise
      except Exception as exc:  # pylint: disable=broad-exception-caught
        last_exc = exc
        if attempt < self.max_retries:
          await asyncio.sleep(0.2 * (2**attempt))
    assert last_exc is not None
    raise last_exc

  async def tag_trajectory_segments(
      self,
      problem: str,
      segments_by_turn: Sequence[Sequence[Segment]],
      test_results: Sequence[TestResult],
      *,
      max_faulty_segments_per_turn: int = 1,
  ) -> dict[str, object]:
    prompt = prompts_lib.build_segment_tag_prompt(
        problem,
        segments_by_turn,
        test_results,
        max_faulty_segments_per_turn=max_faulty_segments_per_turn,
    )
    res = await self._chat_json(
        prompt,
        prompts_lib.SEGMENT_TAG_JSON_SCHEMA,
        system_prompt=prompts_lib.SEGMENT_TAG_SYSTEM_PROMPT,
    )
    if (
        "segments" not in res
        and "tags" in res
        and isinstance(res["tags"], list)
    ):
      res["segments"] = [
          {
              "id": int(t["id"]),
              "label": (
                  str(t["label"])
                  if "label" in t
                  else (
                      "faulty"
                      if ("faulty" in t and bool(t["faulty"]))
                      else "useful"
                  )
              ),
              "reason": str(t["reason"]) if "reason" in t else "OK",
          }
          for t in res["tags"]
          if isinstance(t, dict) and "id" in t
      ]
    if "turn_summary" not in res:
      res["turn_summary"] = []
    return res

  async def audit_group_solutions(
      self,
      problem: str,
      passing_solutions: dict[int, str],
  ) -> dict[str, object]:
    cached = self._audit_cache.get(problem, passing_solutions)
    if cached is not None:
      return cached
    prompt = prompts_lib.build_group_audit_prompt(problem, passing_solutions)
    res = await self._chat_json(prompt, prompts_lib.GROUP_AUDIT_JSON_SCHEMA)
    self._audit_cache.set(problem, passing_solutions, res)
    return res


class MockJudgeClient:
  """Deterministic JudgeClient for unit testing and offline verification."""

  def __init__(
      self,
      *,
      hack_ids: set[int] | None = None,
      faulty_segment_reasons: dict[int, str] | None = None,
      ranking_override: list[int] | None = None,
      fail_mode: Literal[
          "none", "timeout", "malformed_json", "http_error"
      ] = "none",
      latency_s: float = 0.001,
  ) -> None:
    self.hack_ids = set(hack_ids or set())
    self.faulty_segment_reasons = dict(faulty_segment_reasons or {})
    self.ranking_override = ranking_override
    self.fail_mode = fail_mode
    self.latency_s = latency_s
    self.tag_call_count = 0
    self.audit_call_count = 0
    self._audit_cache = AuditCache()

  async def tag_trajectory_segments(
      self,
      problem: str,
      segments_by_turn: Sequence[Sequence[Segment]],
      test_results: Sequence[TestResult],
      *,
      max_faulty_segments_per_turn: int = 1,
  ) -> dict[str, object]:
    del problem, max_faulty_segments_per_turn
    self.tag_call_count += 1
    if self.latency_s > 0:
      await asyncio.sleep(self.latency_s)
    if self.fail_mode == "timeout":
      raise TimeoutError("Judge tag call timed out")
    if self.fail_mode == "malformed_json":
      raise ValueError("Invalid JSON from judge")
    if self.fail_mode == "http_error":
      raise RuntimeError("HTTP 503 Service Unavailable")

    seg_out: list[schemas.SegmentTagDict] = []
    turn_summary: list[schemas.TurnSummaryDict] = []
    for turn_idx, segs in enumerate(segments_by_turn):
      turn_passed = (
          turn_idx < len(test_results) and test_results[turn_idx].all_passed
      )
      for seg in segs:
        seg_body = seg.text or seg.text_preview
        if seg.seg_id in self.faulty_segment_reasons:
          reason = self.faulty_segment_reasons[seg.seg_id]
          label = "useful" if reason == "OK" else "faulty"
        elif not turn_passed and (
            "TODO" in seg_body
            or "wrong" in seg_body.lower()
            or "guess" in seg_body.lower()
            or "ignore" in seg_body.lower()
        ):
          label = "faulty"
          reason = (
              "IGNORED_TRACEBACK" if turn_idx > 0 else "HALLUCINATED_ANALYSIS"
          )
        else:
          label = "useful"
          reason = "OK"
        seg_out.append({"id": seg.seg_id, "label": label, "reason": reason})
      turn_summary.append(
          {"turn": turn_idx + 1, "root_cause_identified": bool(turn_passed)}
      )
    return {"segments": seg_out, "turn_summary": turn_summary}

  async def audit_group_solutions(
      self,
      problem: str,
      passing_solutions: dict[int, str],
  ) -> dict[str, object]:
    if self.latency_s > 0:
      await asyncio.sleep(self.latency_s)
    if self.fail_mode == "timeout":
      raise TimeoutError("Judge audit call timed out")
    if self.fail_mode == "malformed_json":
      raise ValueError("Invalid JSON from judge")
    if self.fail_mode == "http_error":
      raise RuntimeError("HTTP 503 Service Unavailable")

    self.audit_call_count += 1
    if self.ranking_override is None and not self.hack_ids:
      cached = self._audit_cache.get(problem, passing_solutions)
      if cached is not None:
        return cached

    hacks: list[schemas.HackRecordDict] = []
    clean_ids: list[int] = []
    scores: dict[str, dict[str, int]] = {}
    for tid, code in passing_solutions.items():
      is_static_hack, static_reason = detect_static_ast_hacks(code)
      if tid in self.hack_ids or is_static_hack:
        hacks.append({
            "id": tid,
            "reason": static_reason if is_static_hack else "hardcoded_cases",
        })
      else:
        clean_ids.append(tid)
        scores[str(tid)] = {
            "algorithmic_optimality": 5 if len(code) < 200 else 4,
            "edge_case_precision": 5,
            "minimality": 5 if len(code) < 150 else 3,
            "no_dead_code": 5,
            "idiomatic": 4,
        }

    if self.ranking_override is not None:
      ranking = [tid for tid in self.ranking_override if tid in clean_ids]
    else:
      ranking = sorted(
          clean_ids, key=lambda idx: (len(passing_solutions[idx]), idx)
      )

    result: dict[str, object] = {
        "hacks": hacks,
        "scores": scores,
        "ranking": ranking,
    }
    self._audit_cache.set(problem, passing_solutions, result)
    return result
