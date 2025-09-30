# Use official Python 3.13 image from Docker Hub
FROM python:3.13

# Set working directory
WORKDIR /app

# Install Node.js and npm from Debian repositories
RUN apt-get update \
    && apt-get install -y nodejs npm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.in-project true

# Install MCP filesystem server globally
RUN npm install -g @modelcontextprotocol/server-filesystem
