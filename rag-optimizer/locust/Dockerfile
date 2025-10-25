FROM python:3.11-slim

WORKDIR /locust

COPY . /locust

# Install Locust only (it now includes /metrics endpoint natively)
RUN pip install --no-cache-dir locust

EXPOSE 8089

CMD ["locust", "-f", "locustfile.py", "--web-host", "0.0.0.0"]
