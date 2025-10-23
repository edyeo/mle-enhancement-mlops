# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies including poetry
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install poetry

# Configure poetry
RUN poetry config virtualenvs.create false

# Copy poetry files first for better caching
COPY pyproject.toml poetry.lock ./

# Install Python dependencies
RUN poetry install --without dev

# Copy component code
COPY src/ml_pipeline/components/ ./components/
COPY src/ml_pipeline/config/ ./config/

# Set environment variables
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config/pipeline_config.yaml

# Create data directories
RUN mkdir -p /data /output

# Default command (can be overridden in Kubeflow)
CMD ["python", "-c", "print('MLOps Pipeline Component Ready')"]
