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

"""Asynchronous bounded-concurrency group post-processing stage."""

import asyncio
from collections.abc import AsyncIterable, AsyncIterator
import contextlib
from typing import Protocol, runtime_checkable

from absl import logging
from tunix.rl.agentic.agents import agent_types


@runtime_checkable
class GroupPostProcessor(Protocol):
  """Asynchronous post-processor over a prompt group of G trajectories."""

  async def process_group(
      self,
      group: list[agent_types.TrajectoryItem],
  ) -> list[agent_types.TrajectoryItem]:
    """Post-processes a single group of trajectories and returns the group."""
    ...


class BoundedGroupStage:
  """Bounded-concurrency async stream stage wrapping a GroupPostProcessor."""

  def __init__(
      self,
      processor: GroupPostProcessor,
      *,
      max_in_flight: int = 8,
      timeout_s: float = 90.0,
      on_error: str = "passthrough",
  ) -> None:
    if max_in_flight < 1:
      raise ValueError(f"max_in_flight must be >= 1, got {max_in_flight}.")
    if timeout_s <= 0:
      raise ValueError(f"timeout_s must be > 0, got {timeout_s}.")
    if on_error not in ("passthrough", "raise"):
      raise ValueError(
          f"on_error must be 'passthrough' or 'raise', got {on_error!r}."
      )
    self.processor = processor
    self.max_in_flight = max_in_flight
    self.timeout_s = timeout_s
    self.on_error = on_error
    self.fallback_count: int = 0
    self.error_count: int = 0
    self.processed_count: int = 0

  async def _run_one(
      self,
      group: list[agent_types.TrajectoryItem],
  ) -> list[agent_types.TrajectoryItem]:
    try:
      result = await asyncio.wait_for(
          self.processor.process_group(group),
          timeout=self.timeout_s,
      )
      self.processed_count += 1
      return result
    except asyncio.CancelledError:
      raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
      self.error_count += 1
      self.fallback_count += 1
      if self.on_error == "raise":
        raise
      logging.warning(
          "BoundedGroupStage fallback to passthrough after %s: %s",
          type(exc).__name__,
          exc,
      )
      return group

  async def wrap(
      self,
      group_stream: AsyncIterable[list[agent_types.TrajectoryItem]],
  ) -> AsyncIterator[list[agent_types.TrajectoryItem]]:
    """Consumes groups from `group_stream` with bounded concurrency without blocking finished tasks on upstream waits."""
    in_flight: set[asyncio.Task[list[agent_types.TrajectoryItem]]] = set()
    next_group_task: asyncio.Task[list[agent_types.TrajectoryItem]] | None = None
    stream_exhausted = False
    try:
      async with contextlib.aclosing(group_stream.__aiter__()) as aiter:
        while not stream_exhausted or in_flight:
          if (
              not stream_exhausted
              and next_group_task is None
              and len(in_flight) < self.max_in_flight
          ):
            next_group_task = asyncio.create_task(aiter.__anext__())

          wait_set: set[asyncio.Task[list[agent_types.TrajectoryItem]]] = set(
              in_flight
          )
          if (
              next_group_task is not None
              and len(in_flight) < self.max_in_flight
          ):
            wait_set.add(next_group_task)

          if not wait_set:
            break

          done, _ = await asyncio.wait(
              wait_set, return_when=asyncio.FIRST_COMPLETED
          )

          if next_group_task is not None and next_group_task in done:
            done.discard(next_group_task)
            try:
              group = next_group_task.result()
              next_group_task = None
              in_flight.add(asyncio.create_task(self._run_one(group)))
            except StopAsyncIteration:
              next_group_task = None
              stream_exhausted = True

          for finished_task in done:
            in_flight.discard(finished_task)
            yield finished_task.result()
    finally:
      if next_group_task is not None and not next_group_task.done():
        next_group_task.cancel()
      for pending in in_flight:
        if not pending.done():
          pending.cancel()
      all_pending = list(in_flight) + (
          [next_group_task] if next_group_task is not None else []
      )
      if all_pending:
        await asyncio.gather(*all_pending, return_exceptions=True)

  wrap._is_v2_nonblocking_drain = True  # type: ignore[attr-defined]

  def __call__(
      self,
      group_stream: AsyncIterable[list[agent_types.TrajectoryItem]],
  ) -> AsyncIterator[list[agent_types.TrajectoryItem]]:
    return self.wrap(group_stream)
