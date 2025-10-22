# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY cli.py .

# Set environment variables
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config/training_config.yaml

# Create data directory
RUN mkdir -p /data

# Make CLI executable
RUN chmod +x cli.py

# Default command (can be overridden in Kubeflow)
CMD ["python", "cli.py", "train", "--config", "/app/config/training_config.yaml", "--output", "/app/output"]
