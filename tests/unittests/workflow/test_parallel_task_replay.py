# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay of owned tasks whose agent transfers cross workflow schedulers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from google.adk.agents.context import Context
from google.adk.agents.llm_agent import LlmAgent
from google.adk.apps.app import App
from google.adk.apps.app import ResumabilityConfig
from google.adk.events.event import Event
from google.adk.events.event import NodeInfo
from google.adk.events.event_actions import EventActions
from google.adk.events.request_input import RequestInput
from google.adk.tools.long_running_tool import LongRunningFunctionTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import node
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.adk.workflow._errors import NodeInterruptedError
from google.adk.workflow.utils._replay_manager import ReplayManager
from google.genai import types
import pytest

from . import testing_utils


@pytest.mark.parametrize('is_resumable', [False, True])
@pytest.mark.parametrize('nested_group', [False, True])
async def test_cancelled_transfer_to_foreign_scheduler_does_not_block_resume(
    is_resumable: bool,
    nested_group: bool,
):
  """A cancelled parent transfer cannot retain an unrelated replay barrier.

  The outer agent calls an inner workflow through a tool. Its child worker
  transfers back to that outer agent, crossing from the inner scheduler to the
  outer scheduler, and pauses for input. Another worker wins. Later resuming
  an unrelated input request must skip the cancelled hop and completed work.
  """
  chosen = []
  cancelled_ids = []

  def abandoned_approval() -> None:
    """An approval requested by the losing transferred worker."""

  child_model = testing_utils.MockModel.create([
      types.Part.from_function_call(
          name='transfer_to_agent', args={'agent_name': 'outer_agent'}
      ),
  ])
  child_agent = LlmAgent(name='child_agent', model=child_model)

  @node(rerun_on_resume=True)
  async def nested(ctx: Context, node_input: str) -> str:
    async with ctx.parallel_tasks() as tasks:
      return await tasks.spawn(child_agent, node_input).result()

  @node(rerun_on_resume=True)
  async def coordinate(ctx: Context) -> str:
    async with ctx.parallel_tasks() as tasks:
      losing = tasks.spawn(nested if nested_group else child_agent, 'work')

      async def accepted() -> str:
        try:
          await losing.result()
        except NodeInterruptedError:
          return 'winner'
        raise AssertionError('The transferred worker should request approval.')

      tasks.spawn(accepted)
      winner = await tasks.first()
      chosen.append(winner.output)
      cancelled_ids.append(losing.id)
      assert losing.status == 'cancelled'
      return winner.output

  inner = Workflow(name='inner', edges=[(START, coordinate)])

  async def run_inner(tool_context: ToolContext) -> str:
    """Run the inner collaboration workflow."""
    return await tool_context.run_node(inner, node_input='work')

  outer_model = testing_utils.MockModel.create([
      types.Part.from_function_call(name='run_inner', args={}),
      types.Part.from_function_call(name='abandoned_approval', args={}),
      'outer complete',
  ])
  outer_agent = LlmAgent(
      name='outer_agent',
      model=outer_model,
      sub_agents=[child_agent],
      tools=[run_inner, LongRunningFunctionTool(abandoned_approval)],
  )

  @node(rerun_on_resume=True)
  async def approval(ctx: Context) -> AsyncGenerator[Event, None]:
    if 'keep_approval' not in ctx.resume_inputs:
      yield RequestInput(interrupt_id='keep_approval', message='Continue?')
      return
    yield Event(output='finished')

  workflow = Workflow(name='outer', edges=[(START, outer_agent, approval)])
  runner = testing_utils.InMemoryRunner(
      app=App(
          name='transfer_tasks',
          root_agent=workflow,
          resumability_config=ResumabilityConfig(is_resumable=is_resumable),
      )
  )
  try:
    events = await asyncio.wait_for(runner.run_async('start'), 5)
    assert chosen == ['winner']
    assert len(outer_model.requests) == 3
    cancelled_hops = [
        event
        for event in runner.session.events
        if cancelled_ids[0]
        in (event.custom_metadata or {}).get(
            '__adk_internal_parallel_task_owners', []
        )
        and event.long_running_tool_ids
    ]
    assert cancelled_hops
    assert all(
        not event.node_info.path.startswith(cancelled_ids[0] + '/')
        and event.node_info.path != cancelled_ids[0]
        for event in cancelled_hops
    )
    response = types.Content(
        role='user',
        parts=[
            types.Part.from_function_response(
                name='adk_request_input',
                response={'result': 'approved'},
            )
        ],
    )
    response.parts[0].function_response.id = 'keep_approval'
    resumed = await asyncio.wait_for(
        runner.run_async(
            new_message=response,
            invocation_id=events[0].invocation_id,
        ),
        5,
    )
    assert any(event.output == 'finished' for event in resumed)
    assert chosen == ['winner']
    assert len(outer_model.requests) == 3
    assert len(child_model.requests) == 1
  finally:
    await runner.runner.close()


async def test_cancellation_index_rebuilds_after_rewind():
  """Rewinding the decision permits a new execution of the same task path."""
  agent = LlmAgent(name='wf', model=testing_utils.MockModel.create([]))
  invocation = await testing_utils.create_invocation_context(agent)
  invocation.invocation_id = 'run'
  ctx = Context(invocation, node=agent, run_id='1')
  worker_path = 'wf@1/worker@one'
  invocation.session.events.extend([
      Event(
          author='worker',
          invocation_id='run',
          output='discarded',
          node_info=NodeInfo(path=worker_path),
      ),
      Event(
          author='wf',
          invocation_id='run',
          node_info=NodeInfo(path='wf@1/__parallel_tasks_control@parallel-1'),
          custom_metadata={
              '__adk_internal_parallel_tasks': {
                  'group': 'wf@1/parallel@1',
                  'cancelled': [worker_path],
              }
          },
      ),
  ])
  replay = ReplayManager()
  recovered, sequence = replay.scan_workflow_events(ctx)
  assert 'worker@one' not in recovered
  assert 'worker@one' not in sequence

  invocation.session.events.extend([
      Event(
          author='user',
          invocation_id='rewind',
          actions=EventActions(rewind_before_invocation_id='run'),
      ),
      Event(
          author='worker',
          invocation_id='run',
          output='new execution',
          node_info=NodeInfo(path=worker_path),
          custom_metadata={'parallel_task': {'task_id': []}},
      ),
  ])
  recovered, sequence = replay.scan_workflow_events(ctx)
  assert recovered['worker@one'].output == 'new execution'
  assert sequence == ['worker@one']


@pytest.mark.parametrize('task_id', [{'arbitrary': True}, [], 'wf@1/loser@one'])
async def test_cancellation_filter_ignores_unrelated_public_metadata(task_id):
  """Arbitrary user metadata cannot break replay after a cancellation."""
  agent = LlmAgent(name='wf', model=testing_utils.MockModel.create([]))
  invocation = await testing_utils.create_invocation_context(agent)
  invocation.invocation_id = 'run'
  ctx = Context(invocation, node=agent, run_id='1')
  invocation.session.events.extend([
      Event(
          author='wf',
          invocation_id='run',
          node_info=NodeInfo(path='wf@1/__parallel_tasks_control@parallel-1'),
          custom_metadata={
              '__adk_internal_parallel_tasks': {
                  'group': 'wf@1/parallel@1',
                  'cancelled': ['wf@1/loser@one'],
              }
          },
      ),
      Event(
          author='worker',
          invocation_id='run',
          output='kept',
          node_info=NodeInfo(path='wf@1/worker@one'),
          custom_metadata={'parallel_task': {'task_id': task_id}},
      ),
  ])
  recovered, sequence = ReplayManager().scan_workflow_events(ctx)
  assert recovered['worker@one'].output == 'kept'
  assert sequence == ['worker@one']
