# Copyright 2026 Google LLC
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

"""Helpers for tracking active and finished isolation-scoped runs."""

from __future__ import annotations

from collections.abc import Sequence

from ._abort_events import _is_abort_event
from .event import Event


def _find_finished_scope_runs(
    live_events: Sequence[Event],
) -> set[tuple[str, str]]:
  """Returns ``(isolation_scope, invocation_id)`` runs that have terminated.

  A scoped run ``(scope, invocation_id)`` terminates when it emits a terminal
  ``finish_task`` FunctionResponse (``SUCCESS`` or ``ERROR``), permanently
  closing that run even if post-finish status or state-flush events follow in
  the same invocation, or when its ``invocation_id`` was aborted.

  Keying by ``(scope, invocation_id)`` allows workflow task nodes whose
  ``isolation_scope`` is a node path (such as ``wf@1/intake@1``) to open a
  fresh scoped run on a later invocation in the same session.
  """
  from ..agents.llm.task._finish_task_tool import is_finish_task_terminal_fr

  finished_runs: set[tuple[str, str]] = set()
  aborted_invocations = {
      event.invocation_id for event in live_events if _is_abort_event(event)
  }
  for event in live_events:
    if event.actions and event.actions.compaction:
      continue
    scope = event.isolation_scope
    if not scope:
      continue
    if (
        event.invocation_id in aborted_invocations
        or is_finish_task_terminal_fr(event)
    ):
      finished_runs.add((scope, event.invocation_id))
  return finished_runs
