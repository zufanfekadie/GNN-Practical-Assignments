# Production Dockerfile for GNN Model Serving
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and production model artifacts
COPY assignment2_architectures/ ./assignment2_architectures/
COPY assignment4_mlops/ ./assignment4_mlops/

ENV PYTHONPATH=/app
EXPOSE 8000

# Start FastAPI serving via Uvicorn
CMD ["uvicorn", "assignment4_mlops.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
