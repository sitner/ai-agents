import asyncio
import json
import traceback
from typing import Optional
from langchain_core.messages import AIMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from ..graphs import State, get_last_message


class MCPNode:
    def __init__(self, mcp_client: Optional[MultiServerMCPClient] = None):
        """
        MCP Node for LangGraph integration

        Args:
            mcp_client: Configured MultiServerMCPClient instance
        """
        self.mcp_client = mcp_client

    def __call__(self, state: State) -> State:
        """
        Execute MCP tool calls from LLM response

        Args:
            state: Current graph state containing messages with potential tool calls

        Returns:
            Updated state with tool call results
        """
        # Run async code in sync context
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self._execute_async(state))

    async def _execute_async(self, state: State) -> State:
        """Internal async implementation"""
        if not self.mcp_client:
            return state

        # Get the last message from LLM
        last_message = get_last_message(state)
        if not last_message or not isinstance(last_message, AIMessage):
            return state

        # Check if last message has tool_calls
        tool_calls = getattr(last_message, "tool_calls", None) or []
        if not tool_calls:
            # No tool calls to execute
            return state

        try:
            # Get available MCP tools
            available_tools = await self.mcp_client.get_tools()
            tools_dict = {tool.name: tool for tool in available_tools}

            # Execute each tool call
            for tool_call in tool_calls:
                # Extract tool name, id and arguments
                tool_call_id = tool_call.get("id", "unknown")
                tool_name = tool_call.get("name")
                tool_args_str = tool_call.get("args", {})

                # Parse arguments if string
                if isinstance(tool_args_str, str):
                    tool_args = json.loads(tool_args_str)
                else:
                    tool_args = tool_args_str

                if tool_name in tools_dict:
                    try:
                        # Execute the tool
                        tool = tools_dict[tool_name]
                        print("----- Tool Node -----")
                        print(f"Executing tool '{tool_name}' with args: {tool_args}")
                        tool_result = await tool.ainvoke(tool_args)
                        print(tool_result)

                        # Add tool result as ToolMessage (correct format for LLM)
                        state["messages"].append(ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call_id,
                            name=tool_name
                        ))
                    except Exception as tool_error:
                        # Add error as ToolMessage with error status
                        state["messages"].append(ToolMessage(
                            content=f"Error executing tool: {str(tool_error)}",
                            tool_call_id=tool_call_id,
                            name=tool_name,
                            status="error"
                        ))
                else:
                    # Tool not found - add as ToolMessage with error
                    state["messages"].append(ToolMessage(
                        content=f"Tool '{tool_name}' not found in available MCP tools",
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        status="error"
                    ))

        except Exception as e:
            # Handle MCP errors gracefully - if we can't match to a tool_call_id, log as AIMessage
            error_message = f"MCP Error: {str(e)}\n{traceback.format_exc()}"
            print(f"Critical MCP Error: {error_message}")
            state["messages"].append(AIMessage(content=error_message))

        return state
