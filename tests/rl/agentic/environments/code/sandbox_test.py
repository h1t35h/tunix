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

"""Unit tests for SandboxRunner (Gate G3 Invariant #4)."""

import os
from absl.testing import absltest
from tunix.rl.agentic.environments.code import sandbox


class SandboxTest(absltest.TestCase):

  def test_asserts_pass_and_fail(self):
    runner = sandbox.SandboxRunner(default_timeout_s=2.0)
    code = "def add(a, b):\n    return a + b\n"
    res = runner.run(code, ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"])
    self.assertTrue(res.all_passed)
    self.assertEqual(res.passed_count, 2)
    self.assertEqual(res.total, 2)

    res_fail = runner.run(
        code,
        ["assert add(1, 2) == 3", "assert add(2, 2) == 5"],
    )
    self.assertFalse(res_fail.all_passed)
    self.assertEqual(res_fail.passed_count, 1)
    self.assertEqual(res_fail.total, 2)
    self.assertLen(res_fail.failures, 1)

  def test_stdio_format_normalizes_crlf_and_trailing_spaces(self):
    runner = sandbox.SandboxRunner(stdio_timeout_s=2.0)
    code = (
        "import sys\n"
        "nums = [int(x) for x in sys.stdin.read().split()]\n"
        "print(sum(nums), end='   \\r\\n\\r\\n')\n"
    )
    res = runner.run(
        code,
        {
            "format": "stdio",
            "inputs": ["10 20\n", "3 4 5\n"],
            "outputs": ["30\n", "12"],
        },
    )
    self.assertTrue(res.all_passed)
    self.assertEqual(res.passed_count, 2)

  def test_infinite_loop_terminates_within_timeout(self):
    runner = sandbox.SandboxRunner(default_timeout_s=0.5)
    code = "def loop():\n    while True:\n        pass\n"
    res = runner.run(code, ["assert loop() is None"], timeout=0.5)
    self.assertFalse(res.all_passed)
    self.assertTrue(res.timed_out)

  def test_memory_limit_rlimit_as_enforced(self):
    runner = sandbox.SandboxRunner(default_timeout_s=2.0, memory_limit_mb=128)
    code = "def alloc():\n    x = bytearray(512 * 1024 * 1024)\n    return len(x)\n"
    res = runner.run(code, ["assert alloc() > 0"], timeout=2.0)
    self.assertFalse(res.all_passed)
    self.assertEqual(res.passed_count, 0)

  def test_file_size_limit_rlimit_fsize_enforced(self):
    runner = sandbox.SandboxRunner(default_timeout_s=2.0, max_fsize_mb=1)
    code = (
        "def write_big():\n"
        "    with open('big.bin', 'wb') as f:\n"
        "        f.write(b'a' * (4 * 1024 * 1024))\n"
        "    return True\n"
    )
    res = runner.run(code, ["assert write_big() is True"], timeout=2.0)
    self.assertFalse(res.all_passed)
    self.assertEqual(res.passed_count, 0)

  def test_host_environment_variables_stripped(self):
    os.environ["SANDBOX_SECRET_TEST_KEY"] = "top_secret_value"
    try:
      runner = sandbox.SandboxRunner(default_timeout_s=2.0)
      code = "import os\ndef is_clean():\n    return len(os.environ) == 0\n"
      res = runner.run(code, ["assert is_clean() is True"], timeout=2.0)
      self.assertTrue(res.all_passed)
    finally:
      os.environ.pop("SANDBOX_SECRET_TEST_KEY", None)

  def test_no_fork_runtime_warning_under_jax(self):
    import warnings  # pylint: disable=g-import-not-at-top
    import jax  # pylint: disable=g-import-not-at-top

    _ = jax.devices()
    runner = sandbox.SandboxRunner(default_timeout_s=2.0, max_workers=4)
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always", RuntimeWarning)
      res = runner.run(
          "def f(x):\n    return x + 1\n", ["assert f(1) == 2"], timeout=2.0
      )
    self.assertTrue(res.all_passed)
    fork_warnings = [
        w for w in caught if "os.fork()" in str(w.message)
    ]
    self.assertEmpty(fork_warnings)


if __name__ == "__main__":
  absltest.main()
