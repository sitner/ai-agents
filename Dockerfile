# Use official Python 3.13 image from Docker Hub
FROM python:3.13

# Set working directory
WORKDIR /app

# Install Poetry
RUN pip install --upgrade pip \
    && pip install poetry \
    && poetry config virtualenvs.in-project true
