#!/bin/sh
# Start the FastAPI app (uvicorn) in background, then run the worker in foreground
echo "Starting LLM generator FastAPI (uvicorn) on :8000"
uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1 &

# Wait for local FastAPI to become healthy before starting the worker (avoid races)
HEALTH_URL="http://127.0.0.1:8000/health"
RETRIES=15
SLEEP=1
echo "Waiting for $HEALTH_URL to respond..."
count=0
while [ $count -lt $RETRIES ]; do
	if curl -s -f "$HEALTH_URL" > /dev/null 2>&1; then
		echo "Health check passed"
		break
	fi
	count=$((count+1))
	echo "Health check not ready, waiting ${SLEEP}s (attempt ${count}/${RETRIES})"
	sleep $SLEEP
done
if [ $count -ge $RETRIES ]; then
	echo "Warning: FastAPI did not become healthy after ${RETRIES} attempts; starting worker anyway"
fi

echo "Starting LLM generator worker"
python worker.py
