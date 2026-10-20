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

"""Token-to-character offset alignment and turn segmentation."""

import ast
from collections.abc import Sequence
import re
from typing import Any, Literal
import numpy as np
from tunix.rl.agentic.environments.code import schemas

Segment = schemas.Segment

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def compute_token_char_offsets(
    token_ids: Sequence[int] | np.ndarray,
    tokenizer: Any,
) -> tuple[list[tuple[int, int]], str, bool]:
  """Computes exact sampled-token-ID to char offsets with UTF-8 BPE merge recovery."""
  ids = [int(x) for x in token_ids]
  if not ids:
    return [], "", False

  full_text = str(tokenizer.decode(ids, skip_special_tokens=False))
  pieces = [
      str(tokenizer.decode([tid], skip_special_tokens=False)) for tid in ids
  ]

  if "".join(pieces) == full_text:
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for p in pieces:
      nxt = cursor + len(p)
      offsets.append((cursor, nxt))
      cursor = nxt
    return offsets, full_text, False

  offsets = [(0, 0)] * len(ids)
  cursor = 0
  n = len(ids)
  i = 0
  while i < n:
    if "\ufffd" not in pieces[i] and full_text.startswith(pieces[i], cursor):
      plen = len(pieces[i])
      offsets[i] = (cursor, cursor + plen)
      cursor += plen
      i += 1
      continue

    matched = False
    max_span = min(n, i + 8)
    for j in range(i + 2, max_span + 1):
      merged = str(tokenizer.decode(ids[i:j], skip_special_tokens=False))
      if "\ufffd" not in merged and full_text.startswith(merged, cursor):
        mlen = len(merged)
        for k in range(i, j):
          offsets[k] = (cursor, cursor + mlen)
        cursor += mlen
        i = j
        matched = True
        break

    if not matched:
      fallback_offsets = [(0, len(full_text)) for _ in ids]
      return fallback_offsets, full_text, True

  if cursor != len(full_text):
    fallback_offsets = [(0, len(full_text)) for _ in ids]
    return fallback_offsets, full_text, True

  return offsets, full_text, False


def spans_to_token_flags(
    num_tokens: int,
    token_offsets: Sequence[tuple[int, int]],
    flagged_char_spans: Sequence[tuple[int, int]],
) -> np.ndarray:
  """Marks token t with 1 (int8) iff its char interval intersects any flagged span."""
  flags = np.zeros(num_tokens, dtype=np.int8)
  if num_tokens == 0 or not flagged_char_spans:
    return flags
  for t in range(min(num_tokens, len(token_offsets))):
    t_start, t_end = token_offsets[t]
    if t_end <= t_start:
      continue
    for s_start, s_end in flagged_char_spans:
      if max(t_start, s_start) < min(t_end, s_end):
        flags[t] = 1
        break
  return flags


def segment_turn(
    token_ids: Sequence[int] | np.ndarray,
    tokenizer: Any,
    *,
    min_segment_chars: int = 80,
    max_segments: int = 40,
) -> tuple[list[Segment], bool]:
  """Segments a turn into `(think, code, post)` segments with `<80` char merge and `<=40` cap."""
  offsets, full_text, used_fallback = compute_token_char_offsets(
      token_ids, tokenizer
  )
  n_toks = len(token_ids)
  if n_toks == 0 or not full_text:
    return [], used_fallback

  if used_fallback:
    return [
        Segment(
            seg_id=0,
            kind="code",
            char_span=(0, len(full_text)),
            tok_span=(0, n_toks),
            text_preview=full_text[:120],
            text=full_text,
        )
    ], True

  raw_chunks: list[tuple[Literal["think", "code", "post"], int, int]] = []
  ast_spans: list[tuple[int, int]] = []

  think_end = 0
  think_match = re.search(r"<think>(.*?)</think>", full_text, flags=re.DOTALL)
  if think_match:
    t_start, t_end = think_match.span(0)
    think_end = t_end
    parts = [
        m.start()
        for m in re.finditer(
            r"\n\s*\n|(?=Wait,|Let's check|Step \d+)", full_text[t_start:t_end]
        )
    ]
    cuts = sorted(
        set([t_start] + [t_start + p for p in parts if p > 0] + [t_end])
    )
    for a, b in zip(cuts[:-1], cuts[1:]):
      if b > a:
        raw_chunks.append(("think", a, b))

  fence_matches = list(_FENCE_RE.finditer(full_text[think_end:]))
  if fence_matches:
    last_fence = fence_matches[-1]
    f_start = think_end + last_fence.start(0)
    f_end = think_end + last_fence.end(0)
    code_start = think_end + last_fence.start(1)
    code_end = think_end + last_fence.end(1)
    if f_start > think_end:
      raw_chunks.append(("think", think_end, f_start))

    code_body = full_text[code_start:code_end]
    code_lines = code_body.splitlines(keepends=True)
    line_starts = [code_start]
    for ln in code_lines:
      line_starts.append(line_starts[-1] + len(ln))

    try:
      tree = ast.parse(code_body)
      for node in tree.body:
        if not isinstance(node, ast.stmt):
          continue
        ln_s = node.lineno
        ln_e = node.end_lineno if node.end_lineno is not None else ln_s
        c_s = line_starts[max(0, ln_s - 1)]
        c_e = line_starts[min(len(line_starts) - 1, ln_e)]
        if c_e <= c_s:
          continue
        if (
            isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and (c_e - c_s) > 160
            and len(node.body) > 1
        ):
          first_child = node.body[0]
          first_child_ln = max(0, first_child.lineno - 1)
          header_end = line_starts[first_child_ln]
          if header_end > c_s:
            ast_spans.append((c_s, header_end))
            cur_s = header_end
          else:
            cur_s = c_s
          for child in node.body:
            ch_ln_e = (
                child.end_lineno
                if child.end_lineno is not None
                else child.lineno
            )
            ch_e = line_starts[min(len(line_starts) - 1, ch_ln_e)]
            if (ch_e - cur_s) >= 48:
              ast_spans.append((cur_s, ch_e))
              cur_s = ch_e
          if cur_s < c_e:
            if ast_spans and ast_spans[-1][0] >= c_s:
              ast_spans[-1] = (ast_spans[-1][0], c_e)
            else:
              ast_spans.append((cur_s, c_e))
        else:
          ast_spans.append((c_s, c_e))
    except SyntaxError:
      pass

    if ast_spans:
      cursor = f_start
      for c_s, c_e in ast_spans:
        start_pos = min(cursor, c_s)
        raw_chunks.append(("code", start_pos, c_e))
        cursor = c_e
      if cursor < f_end:
        if raw_chunks and raw_chunks[-1][0] == "code":
          prev_k, prev_s, _ = raw_chunks[-1]
          raw_chunks[-1] = (prev_k, prev_s, f_end)
        else:
          raw_chunks.append(("code", cursor, f_end))
    else:
      raw_chunks.append(("code", f_start, f_end))

    if f_end < len(full_text) and full_text[f_end:].strip():
      raw_chunks.append(("post", f_end, len(full_text)))
  else:
    if think_end < len(full_text):
      kind: Literal["think", "code", "post"] = (
          "think" if think_end == 0 else "post"
      )
      raw_chunks.append((kind, think_end, len(full_text)))

  if not raw_chunks:
    raw_chunks.append(("code", 0, len(full_text)))

  code_min_chars = (
      min(min_segment_chars, 48) if len(ast_spans) > 1 else min_segment_chars
  )

  merged: list[tuple[Literal["think", "code", "post"], int, int]] = []
  for kind, s, e in raw_chunks:
    min_chars = code_min_chars if kind == "code" else min_segment_chars
    if (
        merged
        and merged[-1][0] == kind
        and ((e - s) < min_chars or (merged[-1][2] - merged[-1][1]) < min_chars)
    ):
      merged[-1] = (merged[-1][0], merged[-1][1], e)
    else:
      merged.append((kind, s, e))

  i = 0
  while i < len(merged) and len(merged) > 1:
    kind, s, e = merged[i]
    min_chars = code_min_chars if kind == "code" else min_segment_chars
    if (e - s) < min_chars:
      if i + 1 < len(merged) and merged[i + 1][0] == kind:
        merged[i + 1] = (kind, s, merged[i + 1][2])
        merged.pop(i)
        continue
      if i > 0 and merged[i - 1][0] == kind:
        merged[i - 1] = (kind, merged[i - 1][1], e)
        merged.pop(i)
        continue
    i += 1

  while len(merged) > max_segments:
    best_idx = 0
    best_score = float("inf")
    for idx in range(len(merged) - 1):
      combined_len = merged[idx + 1][2] - merged[idx][1]
      penalty = 0 if merged[idx][0] == merged[idx + 1][0] else 100000
      if combined_len + penalty < best_score:
        best_score = combined_len + penalty
        best_idx = idx
    k1, s1, _ = merged[best_idx]
    _, _, e2 = merged[best_idx + 1]
    merged[best_idx : best_idx + 2] = [(k1, s1, e2)]

  segments: list[Segment] = []
  for seg_id, (kind, c_s, c_e) in enumerate(merged):
    overlapping_toks = [
        t
        for t, (t_s, t_e) in enumerate(offsets)
        if max(t_s, c_s) < min(t_e, c_e)
    ]
    if overlapping_toks:
      tok_span = (overlapping_toks[0], overlapping_toks[-1] + 1)
    else:
      tok_span = (0, 0)
    seg_text = full_text[c_s:c_e]
    segments.append(
        Segment(
            seg_id=seg_id,
            kind=kind,
            char_span=(c_s, c_e),
            tok_span=tok_span,
            text_preview=seg_text[:120],
            text=seg_text,
        )
    )
  return segments, False
