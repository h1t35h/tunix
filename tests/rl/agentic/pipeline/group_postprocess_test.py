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

import asyncio
from collections.abc import AsyncIterator
from absl.testing import absltest
from tunix.rl.agentic.agents import agent_types
from tunix.rl.agentic.pipeline import group_postprocess


def _make_group(group_id: int, delay_s: float = 0.0) -> list[agent_types.TrajectoryItem]:
  return [
      agent_types.TrajectoryItem(
          traj={"group_id": group_id, "delay_s": delay_s, "trajectory_reward": 1.0},
          group_id=group_id,
      )
  ]


async def _list_to_async_iter(
    groups: list[list[agent_types.TrajectoryItem]],
) -> AsyncIterator[list[agent_types.TrajectoryItem]]:
  for g in groups:
    yield g


class _MockProcessor:

  def __init__(self, *, fail_group_ids: set[int] | None = None):
    self.fail_group_ids = fail_group_ids or set()
    self.current_in_flight = 0
    self.peak_in_flight = 0

  async def process_group(
      self, group: list[agent_types.TrajectoryItem]
  ) -> list[agent_types.TrajectoryItem]:
    self.current_in_flight += 1
    self.peak_in_flight = max(self.peak_in_flight, self.current_in_flight)
    try:
      gid = group[0].traj["group_id"]
      delay = float(group[0].traj["delay_s"])
      if delay > 0:
        await asyncio.sleep(delay)
      if gid in self.fail_group_ids:
        raise RuntimeError(f"Simulated failure for group {gid}")
      group[0].traj["postprocessed"] = True
      return group
    finally:
      self.current_in_flight -= 1


class GroupPostProcessTest(absltest.TestCase):

  def test_bounded_concurrency_and_out_of_order_completion(self):
    processor = _MockProcessor()
    self.assertIsInstance(processor, group_postprocess.GroupPostProcessor)
    stage = group_postprocess.BoundedGroupStage(
        processor, max_in_flight=2, timeout_s=5.0
    )
    # Group 0 is slow (0.06s), Group 1 is fast (0.005s); with max_in_flight=2,
    # Group 1 should complete and be yielded before Group 0!
    groups = [
        _make_group(0, delay_s=0.06),
        _make_group(1, delay_s=0.005),
        _make_group(2, delay_s=0.005),
        _make_group(3, delay_s=0.005),
    ]

    async def _run():
      out_ids = []
      async for g in stage(_list_to_async_iter(groups)):
        self.assertTrue(g[0].traj["postprocessed"])
        out_ids.append(g[0].traj["group_id"])
      return out_ids

    completed_order = asyncio.run(_run())
    self.assertEqual(sorted(completed_order), [0, 1, 2, 3])
    # Group 1 completes before slow Group 0
    self.assertEqual(completed_order[0], 1)
    self.assertLessEqual(processor.peak_in_flight, 2)
    self.assertEqual(stage.processed_count, 4)
    self.assertEqual(stage.fallback_count, 0)

  def test_timeout_and_exception_fallback_passthrough(self):
    processor = _MockProcessor(fail_group_ids={1})
    stage = group_postprocess.BoundedGroupStage(
        processor, max_in_flight=3, timeout_s=0.02, on_error="passthrough"
    )
    groups = [
        _make_group(0, delay_s=0.001),  # succeeds
        _make_group(1, delay_s=0.001),  # raises RuntimeError -> passthrough
        _make_group(2, delay_s=0.10),   # times out -> passthrough
    ]

    async def _run():
      results = {}
      async for g in stage.wrap(_list_to_async_iter(groups)):
        results[g[0].traj["group_id"]] = g[0].traj
      return results

    res = asyncio.run(_run())
    self.assertEqual(set(res.keys()), {0, 1, 2})
    self.assertTrue(res[0].get("postprocessed", False))
    self.assertFalse(res[1].get("postprocessed", False))
    self.assertFalse(res[2].get("postprocessed", False))
    self.assertEqual(stage.processed_count, 1)
    self.assertEqual(stage.fallback_count, 2)

  def test_nonblocking_drain_while_upstream_waits(self):
    processor = _MockProcessor()
    stage = group_postprocess.BoundedGroupStage(
        processor, max_in_flight=2, timeout_s=2.0
    )

    async def _run():
      first_consumed = asyncio.Event()

      async def _gated_upstream() -> AsyncIterator[list[agent_types.TrajectoryItem]]:
        # Yield group 0 immediately, then block until consumer signals it received group 0.
        yield _make_group(0, delay_s=0.005)
        await asyncio.wait_for(first_consumed.wait(), timeout=1.0)
        yield _make_group(1, delay_s=0.005)

      out_ids = []
      async for g in stage.wrap(_gated_upstream()):
        gid = g[0].traj["group_id"]
        out_ids.append(gid)
        if gid == 0:
          first_consumed.set()
      return out_ids

    completed_order = asyncio.run(_run())
    self.assertEqual(completed_order, [0, 1])
    self.assertEqual(stage.processed_count, 2)


if __name__ == "__main__":
  absltest.main()
