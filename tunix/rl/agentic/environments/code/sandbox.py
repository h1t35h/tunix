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

"""Subprocess sandbox runner for executing untrusted Python solutions."""

import ast
from collections.abc import Sequence
import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
from typing import Any
from tunix.rl.agentic.environments.code import schemas

TestFailure = schemas.TestFailure
TestResult = schemas.TestResult


def _normalize_stdio_output(text: str) -> str:
  lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
  stripped = [line.rstrip() for line in lines]
  while stripped and not stripped[-1]:
    stripped.pop()
  return "\n".join(stripped)


def _strip_temp_paths(tb_text: str, temp_dir: str) -> str:
  cleaned = tb_text.replace(temp_dir, "<sandbox>")
  cleaned = re.sub(
      r'File "/tmp/[^"]+"', 'File "<sandbox>/solution.py"', cleaned
  )
  lines = [ln for ln in cleaned.splitlines() if ln.strip()]
  if len(lines) > 15:
    lines = lines[-15:]
  return "\n".join(lines)


class SandboxRunner:
  """Subprocess sandbox runner (`python -I -S`, rlimits, isolated temp dir)."""

  def __init__(
      self,
      max_workers: int = 32,
      default_timeout_s: float = 3.0,
      stdio_timeout_s: float = 5.0,
      memory_limit_mb: int = 1024,
      max_nproc: int = 64,
      max_fsize_mb: int = 10,
  ) -> None:
    self.max_workers = max_workers
    self.default_timeout_s = default_timeout_s
    self.stdio_timeout_s = stdio_timeout_s
    self.memory_limit_mb = memory_limit_mb
    self.max_nproc = max_nproc
    self.max_fsize_mb = max_fsize_mb
    self._sem = threading.BoundedSemaphore(max(1, int(max_workers)))

  def _build_rlimit_preamble(self, cpu_seconds: int) -> str:
    mem_bytes = int(self.memory_limit_mb) * 1024 * 1024
    fsize_bytes = int(self.max_fsize_mb) * 1024 * 1024
    nproc = int(self.max_nproc)
    return "\n".join([
        "import os as __os, resource as __resource",
        "__os.environ.clear()",
        (
            "for __res, __lim in ("
            f"(__resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds + 1})), "
            f"(__resource.RLIMIT_AS, ({mem_bytes}, {mem_bytes})), "
            f"(__resource.RLIMIT_FSIZE, ({fsize_bytes}, {fsize_bytes})), "
            f"(__resource.RLIMIT_NPROC, ({nproc}, {nproc}))):"
        ),
        "    try:",
        "        __resource.setrlimit(__res, __lim)",
        "    except (ValueError, OSError):",
        "        pass",
        "del __res, __lim, __resource",
    ])

  def _exec_script(
      self,
      script_path: str,
      tmpdir: str,
      timeout: float,
      stdin_data: str | None = None,
  ) -> subprocess.CompletedProcess[str]:
    with self._sem:
      proc = subprocess.Popen(
          [sys.executable, "-I", "-S", script_path],
          cwd=tmpdir,
          env={},
          stdin=(
              subprocess.PIPE
              if stdin_data is not None
              else subprocess.DEVNULL
          ),
          stdout=subprocess.PIPE,
          stderr=subprocess.PIPE,
          text=True,
          start_new_session=True,
      )
      try:
        stdout, stderr = proc.communicate(input=stdin_data, timeout=timeout)
      except subprocess.TimeoutExpired as exc:
        try:
          os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
          proc.kill()
        out, err = proc.communicate()
        raise subprocess.TimeoutExpired(
            proc.args, timeout, output=out, stderr=err
        ) from exc
      return subprocess.CompletedProcess(
          proc.args, proc.returncode, stdout, stderr
      )

  def run(
      self,
      code: str,
      tests: Sequence[str] | dict[str, Any],
      timeout: float | None = None,
      *,
      entry_point: str | None = None,
  ) -> TestResult:
    """Executes `code` against `tests` (`asserts` or `stdio`) in a sandbox."""
    if isinstance(tests, dict) and (
        "inputs" in tests or tests.get("format") == "stdio"
    ):
      return self._run_stdio(code, tests, timeout)
    assert_list = (
        tests.get("asserts", [])
        if isinstance(tests, dict)
        else list(tests)
    )
    return self._run_asserts(
        code, assert_list, timeout, entry_point=entry_point
    )

  def _run_asserts(
      self,
      code: str,
      assert_list: Sequence[str],
      timeout: float | None,
      *,
      entry_point: str | None = None,
  ) -> TestResult:
    if not assert_list:
      return TestResult(passed_count=0, total=0, error="No tests provided")
    eff_timeout = float(
        timeout if timeout is not None else self.default_timeout_s
    )
    cpu_limit = max(1, int(math.ceil(eff_timeout)))

    with tempfile.TemporaryDirectory(prefix="sandbox_asserts_") as tmpdir:
      harness_lines = [
          self._build_rlimit_preamble(cpu_limit),
          "import sys, traceback",
          code,
      ]
      if entry_point:
        harness_lines.append(f"candidate = {entry_point}")
      harness_lines.extend([
          "__passed = 0",
          "__failures = []",
          f"__tests = {repr(list(assert_list))}",
          "for __idx, __stmt in enumerate(__tests):",
          "    try:",
          "        exec(__stmt, globals())",
          "        __passed += 1",
          "    except BaseException as __exc:",
          "        if len(__failures) < 3:",
          (
              "            __tb = ''.join(traceback.format_exception("
              "type(__exc), __exc, __exc.__traceback__))"
          ),
          (
              "            __failures.append("
              "(f'test_{__idx}', __stmt, repr(__exc), __tb))"
          ),
          (
              "print('__SANDBOX_RESULT__:' + "
              "repr((__passed, len(__tests), __failures)))"
          ),
      ])
      script_path = os.path.join(tmpdir, "solution.py")
      with open(script_path, "w", encoding="utf-8") as f:
        f.write("\n".join(harness_lines))

      try:
        proc = self._exec_script(script_path, tmpdir, eff_timeout)
      except subprocess.TimeoutExpired as exc:
        tb_tail = _strip_temp_paths(str(exc.stderr or ""), tmpdir)
        return TestResult(
            passed_count=0,
            total=len(assert_list),
            failures=[
                TestFailure(
                    name="timeout",
                    expected=f"<= {eff_timeout}s",
                    got="TimeoutExpired",
                    traceback_tail=tb_tail,
                )
            ],
            timed_out=True,
            error="TimeoutExpired",
        )

      marker = "__SANDBOX_RESULT__:"
      for line in reversed(proc.stdout.splitlines()):
        if line.startswith(marker):
          payload = ast.literal_eval(line[len(marker) :])
          passed_count, total_count, raw_failures = payload
          failures = [
              TestFailure(
                  name=str(name),
                  expected=str(exp),
                  got=str(got),
                  traceback_tail=_strip_temp_paths(str(tb), tmpdir),
              )
              for (name, exp, got, tb) in raw_failures[:3]
          ]
          return TestResult(
              passed_count=int(passed_count),
              total=int(total_count),
              failures=failures,
              timed_out=False,
              error="",
          )

      timed_out = proc.returncode in (-24, -9, 137, 152)
      err_text = _strip_temp_paths(proc.stderr or proc.stdout, tmpdir)
      return TestResult(
          passed_count=0,
          total=len(assert_list),
          failures=[
              TestFailure(
                  name="runtime_error",
                  expected="exit 0",
                  got=f"returncode={proc.returncode}",
                  traceback_tail=err_text,
              )
          ],
          timed_out=timed_out,
          error=err_text or f"Process exited with {proc.returncode}",
      )

  def _run_stdio(
      self,
      code: str,
      tests: dict[str, Any],
      timeout: float | None,
  ) -> TestResult:
    inputs = list(tests.get("inputs", []))
    outputs = list(tests.get("outputs", []))
    total = len(inputs)
    if total == 0 or len(outputs) != total:
      return TestResult(
          passed_count=0, total=total, error="Invalid stdio test specification"
      )
    eff_timeout = float(
        timeout if timeout is not None else self.stdio_timeout_s
    )
    cpu_limit = max(1, int(math.ceil(eff_timeout)))

    passed_count = 0
    failures: list[TestFailure] = []
    with tempfile.TemporaryDirectory(prefix="sandbox_stdio_") as tmpdir:
      script_path = os.path.join(tmpdir, "solution.py")
      with open(script_path, "w", encoding="utf-8") as f:
        f.write(self._build_rlimit_preamble(cpu_limit) + "\n" + code)

      for idx, (inp, exp_out) in enumerate(zip(inputs, outputs)):
        try:
          proc = self._exec_script(
              script_path, tmpdir, eff_timeout, stdin_data=str(inp)
          )
        except subprocess.TimeoutExpired as exc:
          tb_tail = _strip_temp_paths(str(exc.stderr or ""), tmpdir)
          failures.append(
              TestFailure(
                  name=f"stdio_{idx}",
                  expected=_normalize_stdio_output(str(exp_out)),
                  got="TimeoutExpired",
                  traceback_tail=tb_tail,
              )
          )
          return TestResult(
              passed_count=passed_count,
              total=total,
              failures=failures[:3],
              timed_out=True,
              error="TimeoutExpired",
          )

        if proc.returncode != 0:
          timed_out = proc.returncode in (-24, -9, 137, 152)
          tb_tail = _strip_temp_paths(proc.stderr or proc.stdout, tmpdir)
          if len(failures) < 3:
            failures.append(
                TestFailure(
                    name=f"stdio_{idx}",
                    expected=_normalize_stdio_output(str(exp_out)),
                    got=f"returncode={proc.returncode}",
                    traceback_tail=tb_tail,
                )
            )
          if timed_out:
            return TestResult(
                passed_count=passed_count,
                total=total,
                failures=failures[:3],
                timed_out=True,
                error=tb_tail or "Resource limit exceeded",
            )
          continue

        norm_got = _normalize_stdio_output(proc.stdout)
        norm_exp = _normalize_stdio_output(str(exp_out))
        if norm_got == norm_exp:
          passed_count += 1
        elif len(failures) < 3:
          failures.append(
              TestFailure(
                  name=f"stdio_{idx}",
                  expected=norm_exp,
                  got=norm_got,
                  traceback_tail="",
              )
          )

    return TestResult(
        passed_count=passed_count,
        total=total,
        failures=failures[:3],
        timed_out=False,
        error="",
    )
