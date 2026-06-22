FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

LABEL maintainer="TOMAS Team"
LABEL description="TOMAS ARC-AGI-3 Solver — Taiyi Mutual-Play framework"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install -e .

# Create directories
RUN mkdir -p /app/logs /app/runs /app/config

# Set environment variables
ENV PYTHONPATH=/app
ENV TOMAS__LOGGING__LOG_DIR=/app/logs

# Default command
CMD ["python", "main.py", "--help"]
