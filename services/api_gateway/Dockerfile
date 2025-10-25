# Use a slim Python image for smaller size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED 1
ENV APP_HOME /app
ENV PORT 8000

# Set the working directory
WORKDIR $APP_HOME

# Copy only the requirements file first to leverage Docker cache
# NOTE: The path is relative to the directory containing the Dockerfile
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app.py .

# Expose the port the app runs on
EXPOSE $PORT

# Command to run the application using Gunicorn with Uvicorn workers (production setup)
# This maximizes throughput by using asynchronous workers.
# The application module is 'app', and the FastAPI object is 'app'.
CMD ["gunicorn", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "app:app"]
