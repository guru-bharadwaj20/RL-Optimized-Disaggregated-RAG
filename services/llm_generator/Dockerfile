# In services/llm_generator/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Copy global requirements from the project root
COPY ../requirements.txt . 

# Install all dependencies 
RUN pip install --no-cache-dir -r requirements.txt 

# Copy service-specific application code
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
