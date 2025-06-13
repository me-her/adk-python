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

from google.adk.agents import Agent
from google.adk.evaluation.agent_creator import IdentityAgentCreator
from google.adk.tools.tool_context import ToolContext
from google.genai import types


def _method_1(arg1: int, tool_context: ToolContext) -> (int, ToolContext):
  return (arg1, tool_context)


async def _method_2(arg1: list[int]) -> list[int]:
  return arg1


_TEST_SUB_AGENT = Agent(
    model="gemini-2.0-flash",
    name="test_sub_agent",
    description="test sub-agent description",
    instruction="test sub-agent instructions",
    tools=[
        _method_1,
        _method_2,
    ],
)

_TEST_AGENT_1 = Agent(
    model="gemini-2.0-flash",
    name="test_agent_1",
    description="test agent description",
    instruction="test agent instructions",
    tools=[
        _method_1,
        _method_2,
    ],
    sub_agents=[_TEST_SUB_AGENT],
    generate_content_config=types.GenerateContentConfig(
        safety_settings=[
            types.SafetySetting(  # avoid false alarm about rolling dice.
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
    ),
)


def test_identity_agent_creator():
  creator = IdentityAgentCreator(root_agent=_TEST_AGENT_1)

  agent1 = creator.get_agent()
  agent2 = creator.get_agent()

  assert isinstance(agent1, Agent)
  assert isinstance(agent2, Agent)

  assert agent1 is not _TEST_AGENT_1  # Ensure it's a copy
  assert agent2 is not _TEST_AGENT_1  # Ensure it's a copy
  assert agent1 is not agent2  # Ensure different copies are returned

  assert agent1.sub_agents[0] is not _TEST_SUB_AGENT  # Ensure it's a copy
  assert agent2.sub_agents[0] is not _TEST_SUB_AGENT  # Ensure it's a copy
  assert (
      agent1.sub_agents[0] is not agent2.sub_agents[0]
  )  # Ensure different copies are returned
