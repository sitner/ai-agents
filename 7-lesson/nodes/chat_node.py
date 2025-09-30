from typing import List
from langchain_core.messages import BaseMessage
from langchain_core.language_models import BaseChatModel
from ..graphs import State


class ChatNode:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def __call__(self, state: State) -> State:
        messages: List[BaseMessage] = state["messages"]

        print("----- Chat Node -----")
        print(f"Messages to LLM: {[msg.model_dump() for msg in messages]}")

        # Call Ollama directly with LangChain messages
        response: BaseMessage = self.llm.invoke(messages)

        print(f"LLM Response: {response}")

        # Append to state
        state["messages"].append(response)
        return state
