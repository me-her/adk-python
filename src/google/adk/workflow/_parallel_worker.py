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

"""Parallel worker node for workflows."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from collections.abc import Iterator
from collections.abc import Mapping
from typing import Any

from google.genai import types
from pydantic import ConfigDict
from pydantic import Field
from pydantic import PrivateAttr
from typing_extensions import override

from ..agents.base_agent import BaseAgent
from ..agents.context import Context
from ..utils._schema_utils import _unwrap_content_payload
from ._base_node import BaseNode
from ._errors import WorkflowConfigurationError
from ._graph import NodeLike
from ._parallel_tasks import ParallelTasks
from ._retry_config import RetryConfig
from .utils._workflow_graph_utils import build_node

# How long to wait for cancelled items to actually stop before giving up on
# them, so an item that swallows cancellation cannot hang this node forever.
_CANCELLED_ITEM_DRAIN_TIMEOUT_SECONDS = 5.0


def _get_node_input_annotation(node: BaseNode) -> Any:
  """Returns the input type annotation for the wrapped node, if any."""
  schema = getattr(node, 'input_schema', None)
  if schema not in (None, Any, object) and not isinstance(schema, dict):
    return schema
  type_hints = getattr(node, '_type_hints', None)
  return type_hints.get('node_input') if isinstance(type_hints, dict) else None


def _unwrap_content_input(
    node_input: types.Content, inner_node: BaseNode
) -> Any:
  """Unwraps text/JSON payload from types.Content for parallel fan-out."""
  return _unwrap_content_payload(
      node_input,
      _get_node_input_annotation(inner_node),
      fan_out_list=True,
  )


def _with_retry_config(
    node: BaseNode, retry_config: RetryConfig | None
) -> BaseNode:
  """Returns a copy of ``node`` that uses ``retry_config``.

  Args:
    node: The node to copy. It is left unchanged.
    retry_config: The retry policy for the copy, replacing the node's own, or
      ``None`` to clear it.

  Returns:
    The copy, of the same class as ``node``.
  """
  copied: BaseNode
  # An agent is cloned rather than copied, so that it gets its own sub-agents
  # and its callbacks are rebound to the copy instead of the original.
  if isinstance(node, BaseAgent):
    copied = node.clone(update={'retry_config': retry_config})
    # BaseAgent.clone() resets parent_agent to None; restore it.
    copied.parent_agent = node.parent_agent
  else:
    copied = node.model_copy(update={'retry_config': retry_config})
  return copied


class _ParallelWorker(BaseNode):
  """A node that runs a wrapped node in parallel for each item in the input list.

  ``retry_config`` applies to each item, never to the worker as a whole. The
  worker moves it onto the wrapped node, replacing any policy the wrapped node
  had, so a failed item is retried in place while its siblings keep running,
  and the worker's own ``retry_config`` stays None. ``timeout`` applies to the
  worker as a whole, so a timeout of the whole worker is not retried.

  Attributes:
    max_parallel_workers: The maximum number of parallel tasks to run. If None,
      there is no limit on concurrency.
  """

  model_config = ConfigDict(arbitrary_types_allowed=True)

  max_parallel_workers: int | None = Field(default=None)

  _node: BaseNode = PrivateAttr()

  def __init__(
      self,
      *,
      node: NodeLike,
      max_parallel_workers: int | None = None,
      retry_config: RetryConfig | None = None,
      timeout: float | None = None,
  ):
    """Initializes the worker.

    Args:
      node: The node to run once per item: a node, an agent, a tool or a
        callable, built into a node with ``build_node``.
      max_parallel_workers: The maximum number of items to run at once. None
        means no limit.
      retry_config: The retry policy for each item. When set, it is moved onto
        the wrapped node, replacing the node's own policy if it has one, just as
        ``node(..., retry_config=...)`` overrides a node's policy. If ``None``,
        the wrapped node keeps its own policy.
      timeout: The time limit for the whole worker, in seconds, covering all
        of its items. ``None`` means no time limit.

    Raises:
      WorkflowConfigurationError: If ``node`` is START, or if
        ``max_parallel_workers`` is less than 1.
    """
    if node == 'START':
      raise WorkflowConfigurationError(
          'ParallelWorker cannot wrap a START node.'
      )
    built_node = build_node(node)
    if retry_config is not None:
      # Moved rather than copied, as node(fn, parallel_worker=True,
      # retry_config=...) does: the policy applies to each item, and a failure
      # of the worker as a whole, such as its timeout, is not retried.
      built_node = _with_retry_config(built_node, retry_config)
    super().__init__(
        name=built_node.name,
        rerun_on_resume=True,
        timeout=timeout,
    )
    if max_parallel_workers is not None and max_parallel_workers < 1:
      raise WorkflowConfigurationError(
          'max_parallel_workers must be greater than or equal to 1.'
      )
    self._node = built_node
    self.max_parallel_workers = max_parallel_workers

  @override
  def _iter_static_children(self) -> Iterator[BaseNode]:
    """Yields the wrapped node."""
    yield self._node

  @override
  def model_copy(
      self, *, update: Mapping[str, Any] | None = None, deep: bool = False
  ) -> _ParallelWorker:
    """Copies the worker, moving any ``retry_config`` onto the wrapped node.

    ``node(worker, retry_config=...)`` reaches the worker through this method,
    so the policy is handled as in ``__init__``: it replaces the wrapped node's
    own and is cleared on the copy, leaving it to apply to each item.

    Args:
      update: Values for fields of the copy, replacing the worker's. If
        ``None``, no field values are replaced.
      deep: Whether to make a deep copy.

    Returns:
      The copy of the worker.
    """
    copied = super().model_copy(update=update, deep=deep)
    if copied.retry_config is not None or (
        update is not None and 'retry_config' in update
    ):
      copied._node = _with_retry_config(copied._node, copied.retry_config)
      copied.retry_config = None
    return copied

  @override
  async def _run_impl(
      self,
      *,
      ctx: Context,
      node_input: Any,
  ) -> AsyncGenerator[Any, None]:
    if isinstance(node_input, types.Content):
      node_input = _unwrap_content_input(node_input, self._node)

    if not isinstance(node_input, list):
      # Wrap the single input in a list to allow processing.
      # This handles cases where the input is a single item.
      node_input = [node_input]

    if not node_input:
      yield []
      return

    async with ParallelTasks(
        ctx,
        name=self.name,
        max_parallel_workers=self.max_parallel_workers,
        cancellation_timeout=_CANCELLED_ITEM_DRAIN_TIMEOUT_SECONDS,
        _legacy_run_ids=True,
    ) as tasks:
      yield await tasks._gather_items(self._node, node_input)
