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

"""Unit tests for segmenter (Gate G3 Invariant #3)."""

from collections.abc import Sequence
from absl.testing import absltest
import numpy as np
from tunix.rl.agentic.environments.code import segmenter


class _ByteLevelTestTokenizer:
  """Byte-level tokenizer that faithfully reproduces UTF-8 multi-byte splits."""

  def encode(self, text: str) -> list[int]:
    return list(text.encode("utf-8"))

  def decode(
      self, token_ids: Sequence[int], skip_special_tokens: bool = False
  ) -> str:
    del skip_special_tokens
    return bytes(int(b) for b in token_ids).decode("utf-8", errors="replace")


class SegmenterTest(absltest.TestCase):

  def test_offset_roundtrip_across_20_diverse_snippets(self):
    """Gate G3 Invariant #3: exact char-offset roundtrip on 20 diverse Python/UTF-8 snippets."""
    tok = _ByteLevelTestTokenizer()
    snippets = [
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        "# Unicode comment: α + β = γ, π ≈ 3.14159\ndef area(r):\n    return 3.14159 * r * r\n",
        "s = 'Emoji test 🚀🔥✨ and CJK 中文测试'\nprint(len(s))\n",
        "<think>\nLet's check edge cases for n=0 and n=1.\n\nWait, n can be negative!\n</think>\n```python\ndef f(n): return abs(n)\n```",
        "def matrix_mul(A, B):\n    return [[sum(a*b for a, b in zip(row, col)) for col in zip(*B)] for row in A]\n",
        "import heapq\ndef dijkstra(g, s):\n    pq = [(0, s)]\n    return pq\n",
        "'''Multi-line docstring with accents: café, naïve, résumé.'''\ndef greet():\n    return 'Hello'\n",
        "x = [i**2 for i in range(10) if i % 2 == 0]\n",
        "try:\n    1 / 0\nexcept ZeroDivisionError as e:\n    print(f'Error: {e}')\n",
        "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        "class Node:\n    def __init__(self, val=0, nxt=None):\n        self.val = val\n        self.nxt = nxt\n",
        "from collections import Counter\ndef anagram(a, b):\n    return Counter(a) == Counter(b)\n",
        "import sys\ndata = sys.stdin.read().splitlines()\nprint('\\n'.join(reversed(data)))\n",
        "def binary_search(arr, target):\n    lo, hi = 0, len(arr) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1\n",
        "math_symbols = '∑ ∏ ∫ √ ∞ ≠ ≤ ≥ ∈ ∉ ⊂ ⊃ ∪ ∩'\n",
        "escaped = 'Line1\\nLine2\\tTabbed\\\\Backslash'\n",
        "lambda_sort = sorted([('b', 2), ('a', 1)], key=lambda p: (p[1], p[0]))\n",
        "async def fetch(x):\n    return x * 2\n",
        "with open('out.txt', 'w', encoding='utf-8') as f:\n    f.write('テスト')\n",
        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
    ]
    self.assertLen(snippets, 20)
    for snippet in snippets:
      ids = tok.encode(snippet)
      offsets, full_text, used_fallback = segmenter.compute_token_char_offsets(
          ids, tok
      )
      self.assertFalse(used_fallback)
      self.assertEqual(full_text, snippet)
      self.assertLen(offsets, len(ids))
      self.assertEqual(offsets[0][0], 0)
      self.assertEqual(offsets[-1][1], len(snippet))
      for (s0, e0), (s1, e1) in zip(offsets[:-1], offsets[1:]):
        self.assertLessEqual(s0, s1)
        self.assertLessEqual(e0, e1)

  def test_segment_turn_splits_think_code_post_and_caps_at_40(self):
    tok = _ByteLevelTestTokenizer()
    steps_text = "\n\n".join(
        [
            f"Step {i}: Analyze subproblem {i} with detailed reasoning and invariant checks across bounds."
            for i in range(55)
        ]
    )
    text = (
        f"<think>\n{steps_text}\n</think>\n"
        "```python\n"
        "def first_helper(x: int) -> int:\n"
        "    total = x + 1\n"
        "    return total * 2\n\n"
        "def second_helper(y: int) -> int:\n"
        "    return first_helper(y) + 10\n"
        "```\n"
        "This solution runs in O(1) time and O(1) auxiliary space across all test cases."
    )
    ids = tok.encode(text)
    segments, used_fb = segmenter.segment_turn(ids, tok, max_segments=40)
    self.assertFalse(used_fb)
    self.assertLessEqual(len(segments), 40)
    kinds = {s.kind for s in segments}
    self.assertIn("think", kinds)
    self.assertIn("code", kinds)
    self.assertIn("post", kinds)

  def test_spans_to_token_flags(self):
    tok = _ByteLevelTestTokenizer()
    text = "hello faulty_part world"
    ids = tok.encode(text)
    offsets, _, _ = segmenter.compute_token_char_offsets(ids, tok)
    flags = segmenter.spans_to_token_flags(len(ids), offsets, [(6, 17)])
    self.assertEqual(flags.dtype, np.int8)
    np.testing.assert_array_equal(flags[:6], 0)
    np.testing.assert_array_equal(flags[6:17], 1)
    np.testing.assert_array_equal(flags[17:], 0)

  def test_segment_turn_splits_long_function_into_statement_blocks(self):
    tok = _ByteLevelTestTokenizer()
    text = (
        "<think>\nWe should process the array in three stages.\n</think>\n"
        "```python\n"
        "def solve_array_problem(values: list[int], threshold: int) -> int:\n"
        "    filtered = [v for v in values if v >= threshold]\n"
        "    if not filtered:\n"
        "        return 0\n"
        "    prefix_sums = [0] * (len(filtered) + 1)\n"
        "    for idx, val in enumerate(filtered):\n"
        "        prefix_sums[idx + 1] = prefix_sums[idx] + val * 2\n"
        "    best_score = max(prefix_sums) - min(prefix_sums)\n"
        "    return int(best_score)\n"
        "```\n"
    )
    ids = tok.encode(text)
    segments, used_fb = segmenter.segment_turn(ids, tok)
    self.assertFalse(used_fb)
    code_segs = [s for s in segments if s.kind == "code"]
    self.assertGreaterEqual(len(code_segs), 3)
    for seg in segments:
      self.assertTrue(seg.text)
      self.assertEqual(seg.text[:120], seg.text_preview)


if __name__ == "__main__":
  absltest.main()
