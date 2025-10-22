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

# Copy component code
COPY components/ ./components/
COPY config/ ./config/

# Set environment variables
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/config/pipeline_config.yaml

# Create data directories
RUN mkdir -p /data /output

# Default command (can be overridden in Kubeflow)
CMD ["python", "-c", "print('MLOps Pipeline Component Ready')"]
