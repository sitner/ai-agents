from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class State(TypedDict):
    """Base state class for LangGraph workflows."""

    messages: Annotated[list[BaseMessage], add_messages]


def get_last_message(state: State) -> Optional[BaseMessage]:
    """
    Get the last message from state.

    Args:
        state: Current graph state

    Returns:
        Last message or None if no messages
    """
    return state["messages"][-1] if state["messages"] else None