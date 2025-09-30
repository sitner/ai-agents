from langchain_core.messages import AIMessage
from .state import State, get_last_message


class RouteTools:
    """Routes based on whether LLM response contains tool calls."""

    def __init__(self, tools_node: str = "tools", end_node: str = "end"):
        """
        Initialize routing logic.

        Args:
            tools_node: Name of the node to route to when tools are needed
            end_node: Name of the node to route to when no tools are needed
        """
        self.tools_node = tools_node
        self.end_node = end_node

    def __call__(self, state: State) -> str:
        """
        Route decision based on tool calls in last message.

        Args:
            state: Current graph state

        Returns:
            Node name to route to
        """
        # Get the last message from LLM
        last_message = get_last_message(state)
        print("----- Router -----")
        print(last_message)

        # Check if it's an AI message with tool calls
        if isinstance(last_message, AIMessage):
            tool_calls = getattr(last_message, "tool_calls", None) or []
            if tool_calls:
                return self.tools_node

        # No tool calls - go to end
        return self.end_node
