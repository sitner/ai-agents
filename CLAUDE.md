# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI agents project repository for learning and developing AI agents. The project uses Poetry for dependency management and Docker for containerized development. It includes implementations using OpenAI API, LangGraph, and LangChain with MCP integration.

## Development Environment

The project uses Docker for development with Python 3.13 as the base runtime environment.

### Docker Setup
- **Image**: Python 3.13 official image
- **Package Manager**: Poetry (installed in container)
- **Container Name**: `ai_agents`
- **Working Directory**: `/app`

## Dependencies

The project uses Poetry with `package-mode = false` for dependency-only management:
- **openai** - OpenAI API client
- **python-dotenv** - Environment variables from .env files
- **litellm** - Multi-LLM API client
- **devtools** - Development tools for debugging
- **langgraph** - Building stateful multi-actor applications with LLMs
- **langchain-mcp-adapters** - MCP (Model Context Protocol) adapters for LangChain
- **langchain-ollama** - Ollama integration for LangChain
- **gymnasium** - Reinforcement learning environments
- **numpy** - Numerical computing for Q-learning
- **matplotlib** - Visualization and plotting

## Project Structure

```
ai-agents/
├── 1-lesson/           # Lesson 1: OpenAI API basics
│   └── main.py        # OpenAI API example script
├── 4-lesson/           # Lesson 4: N8N automation
│   ├── Dockerfile     # N8N custom container with finance packages
│   ├── docker-compose.yml # N8N + Redis orchestration
│   └── 4-lesson.json  # N8N workflow configuration
├── 7-lesson/           # Lesson 7: LangGraph AI agent with MCP
│   ├── agents/        # Agent implementations
│   ├── graphs/        # LangGraph state and routing
│   │   ├── route_tools.py  # Tool routing logic
│   │   └── state.py        # State management
│   ├── nodes/         # LangGraph nodes
│   │   ├── chat_node.py    # Chat node implementation
│   │   └── mcp_node.py     # MCP integration node
│   ├── vault/         # Obsidian vault for testing
│   ├── docker-compose.yml  # MCP server orchestration
│   ├── main.py        # Main LangGraph application
│   └── .env           # Environment configuration
├── 10-lesson/          # Lesson 10: Q-learning agent with GridWorld
│   ├── grid_env.py    # Custom GridWorld environment (Gymnasium)
│   ├── q_learning.py  # Q-learning algorithm implementation
│   ├── main.py        # Training and evaluation pipeline
│   ├── utils.py       # Visualization and policy utilities
│   └── docker-compose.yml  # Container setup for lesson 10
├── .vscode/           # VS Code configuration
│   └── settings.json  # Python interpreter and analysis paths
├── pyproject.toml     # Poetry configuration with dependencies
├── poetry.lock        # Locked dependency versions
├── Dockerfile         # Python 3.13 container setup
├── docker-compose.yml # Python container orchestration
└── README.md          # Project documentation
```

## VS Code Setup

The project is configured for VS Code with Docker:
- Python analysis paths point to Docker container packages
- Auto-save enabled with 1-second delay
- Python interpreter configured for container environment

## Development Commands

### Python Development (Lessons 1-3)

**Install dependencies:**
```bash
docker exec ai_agents poetry install
```

**Run Python scripts:**
```bash
docker exec -it ai_agents python /app/1-lesson/main.py
```

**Run with Poetry:**
```bash
docker exec -it ai_agents poetry run python /app/1-lesson/main.py
```

### N8N Development (Lesson 4)

**Start N8N with Redis:**
```bash
cd 4-lesson
docker-compose up -d
```

**Access N8N interface:**
- URL: http://localhost:5678
- Redis available at localhost:6379

**Stop N8N services:**
```bash
cd 4-lesson
docker-compose down
```

### LangGraph Development (Lesson 7)

**Start MCP server:**
```bash
cd 7-lesson
docker-compose up -d
```

**Run LangGraph agent:**
```bash
docker exec -it ai_agents python -m 7-lesson.main
```

**Environment variables (.env):**
- `VAULT_PATH` - Path to Obsidian vault (default: /app/7-lesson/vault)
- `LLM_MODEL` - Ollama model name (e.g., gpt-oss:20b)
- `OLLAMA_API_BASE` - Ollama API endpoint (default: http://host.docker.internal:11434)

**Stop MCP services:**
```bash
cd 7-lesson
docker-compose down
```

### Q-Learning Development (Lesson 10)

**Start container:**
```bash
cd 10-lesson
docker-compose up -d
```

**Install dependencies:**
```bash
docker exec ai_agents_lesson10 poetry install
```

**Train Q-learning agent:**
```bash
docker exec ai_agents_lesson10 poetry run python main.py
```

**Features:**
- Custom GridWorld environment with configurable size and obstacles
- Q-learning agent with epsilon-greedy exploration
- Random goal position generation (different each run)
- Q-table persistence with goal position tracking
- Real-time ASCII visualization of agent behavior
- Policy evaluation and success rate metrics

**Q-table naming:**
- Format: `q_table_grid{size}_obs{n}_goal{row}_{col}.npy`
- Example: `q_table_grid8_obs0_goal3_5.npy` (8x8 grid, 0 obstacles, goal at (3,5))
- Q-tables are specific to grid configuration and goal position

**Stop container:**
```bash
cd 10-lesson
docker-compose down
```

## Notes for Development

- Dependencies are installed in Docker container, not host system
- VS Code configured to recognize packages in container via `python.analysis.extraPaths`
- Poetry configured with `package-mode = false` for script-only development
- All Python execution should happen within Docker container
- Environment variables should be loaded from .env files using python-dotenv