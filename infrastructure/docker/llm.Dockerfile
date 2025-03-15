FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for model weights
RUN mkdir -p /app/models/llm/weights

# Expose port
EXPOSE 8001

# Command to run the LLM service
CMD ["uvicorn", "models.llm.service:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
