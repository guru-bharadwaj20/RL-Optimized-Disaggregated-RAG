FROM python:3.11-slim
WORKDIR /app

# Install build deps and pip
RUN apt-get update && apt-get install -y --no-install-recommends gcc build-essential git && rm -rf /var/lib/apt/lists/*

COPY services/encoder_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/encoder_service /app

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
# In services/encoder_service/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Copy global requirements from the root context
COPY ../requirements.txt . 

# Install all dependencies at once (simplifies build, but increases image size)
RUN pip install --no-cache-dir -r requirements.txt 

# Copy service-specific application code
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]