import os
import asyncio
from dotenv import load_dotenv
from .nodes import ChatNode, MCPNode
from .graphs import State, RouteTools
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from devtools import debug
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

load_dotenv()

# Vault path configuration
vault_path = os.getenv("VAULT_PATH", "/app/7-lesson/vault")

# System prompt for second brain assistant
system_prompt = f"""You are an intelligent assistant that manages and works with Personal Knowledge Management System (PKMS) in Obsidian.

Your capabilities:
- Analyze and organize markdown files
- Find connections between knowledge and concepts
- Help with linking information using wikilinks [[]]
- Suggest tags and categories for better organization
- Create concept maps and knowledge graphs
- Summarize and prepare topic overviews
- Assist with creating new notes and articles

You have access to filesystem tools that allow you to:
- Read files and directories in the Obsidian vault
- Search for files and patterns
- Create and edit markdown files
- List directory contents

IMPORTANT: Always use the available tools to search and read files in the vault before answering questions about its contents. Don't assume what files exist - use list_directory, search_files, and read_file tools to find the information.

The vault is located at: {vault_path}
When using tools, always use this path or paths within it. For example:
- list_directory: use "{vault_path}"
- search_files: use "{vault_path}" as the path parameter
- read_file: use full path like "{vault_path}/filename.md"

IMPORTANT: After you have gathered the necessary information using tools, ALWAYS provide a response to the user in Czech. Do NOT keep calling the same tool repeatedly. Once you have the information, summarize it and respond.

Always respond in Czech and strive to be as helpful as possible when working with the knowledge system."""

filesystem_config = {
    "filesystem": {
        "command": "mcp-server-filesystem",
        "args": [vault_path],
        "transport": "stdio",
    }
}
mcp_client = MultiServerMCPClient(filesystem_config)

# Get tools async
async def get_mcp_tools():
    return await mcp_client.get_tools()

# Run async function to get tools
all_tools = asyncio.run(get_mcp_tools())

# Initialize Ollama LLM
llm = ChatOllama(
    model=os.getenv("LLM_MODEL", "gemma3:27b"),
    base_url=os.getenv("OLLAMA_API_BASE", "http://host.docker.internal:11434"),
    temperature=0.7,
)

# Bind tools to LLM
llm_with_tools = llm.bind_tools(all_tools)

chat_node = ChatNode(llm_with_tools)

mcp_node = MCPNode(mcp_client=mcp_client)
# Build LangGraph workflow
graph_builder = StateGraph(State)


""" debug(
    client.chat_completion(
        messages=[{"role": "system", "content": "You are a helpful assistant."}]
    )
) """

# Add nodes to graph
graph_builder.add_node("chat", chat_node)
graph_builder.add_node("mcp", mcp_node)
route_tools = RouteTools("mcp", END)

graph_builder.add_edge(START, "chat")
graph_builder.add_conditional_edges(
    "chat",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"mcp": "mcp", END: END},
)
graph_builder.add_edge("mcp", "chat")


# Compile the graph with recursion limit to prevent infinite loops
graph = graph_builder.compile(
    checkpointer=None,  # No checkpointing needed for now
    interrupt_before=None,
    interrupt_after=None,
    debug=False
)


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream(
        State(
            messages=[
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]
        ),
        config={"recursion_limit": 15}  # Limit to 10 iterations to prevent infinite loops
    ):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)
