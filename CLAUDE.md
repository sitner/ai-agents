# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI agents project repository for learning and developing AI agents using OpenAI API. The project uses Poetry for dependency management and Docker for containerized development.

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

## Project Structure

```
ai-agents/
├── 1-lesson/           # Lesson 1: OpenAI API basics
│   └── main.py        # OpenAI API example script
├── 4-lesson/           # Lesson 4: N8N automation
│   ├── Dockerfile     # N8N custom container with finance packages
│   ├── docker-compose.yml # N8N + Redis orchestration
│   └── 4-lesson.json  # N8N workflow configuration
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
docker exec -it ai_agents poetry install
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

## Notes for Development

- Dependencies are installed in Docker container, not host system
- VS Code configured to recognize packages in container via `python.analysis.extraPaths`
- Poetry configured with `package-mode = false` for script-only development
- All Python execution should happen within Docker container
- Environment variables should be loaded from .env files using python-dotenv