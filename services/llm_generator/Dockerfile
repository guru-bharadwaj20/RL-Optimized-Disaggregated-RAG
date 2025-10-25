FROM python:3.11-slim

WORKDIR /app

# Install minimal system deps for common Python packages
RUN apt-get update \
	&& apt-get install -y --no-install-recommends gcc build-essential \
	&& rm -rf /var/lib/apt/lists/*

# Copy and install service-specific requirements (build context is the service folder)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy service code and entrypoint (from build context) into the image
COPY . /app
RUN chmod +x /app/entrypoint.sh

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]
