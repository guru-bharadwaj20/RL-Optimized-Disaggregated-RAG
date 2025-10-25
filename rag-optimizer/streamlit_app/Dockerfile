FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir streamlit pillow

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
