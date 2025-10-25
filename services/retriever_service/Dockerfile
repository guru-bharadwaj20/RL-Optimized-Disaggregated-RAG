FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential && rm -rf /var/lib/apt/lists/*

COPY services/retriever_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/retriever_service /app

ENV PYTHONUNBUFFERED=1
CMD ["python", "worker.py"]
# Use a slim Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
# Set environment variables for services to ensure proper connection from Docker Compose
ENV REDIS_HOST redis
ENV QDRANT_HOST qdrant
ENV QDRANT_PORT 6333

# Set the working directory
WORKDIR $APP_HOME

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the worker script
COPY worker.py .

# Run the worker script
CMD ["python", "worker.py"]
