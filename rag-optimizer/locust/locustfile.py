from locust import HttpUser, task, between
import json

SAMPLE_QUERIES = [
    "What is recursion in computer science?",
    "Explain the difference between REST and GraphQL.",
    "How does machine learning improve RAG systems?",
    "What are the benefits of microservices architecture?",
    "Describe the role of embeddings in vector databases.",
    "Why is caching important in distributed systems?",
    "What is the purpose of a message queue?",
    "How does load balancing work in cloud computing?",
    "Explain the CAP theorem.",
    "What is the difference between SQL and NoSQL databases?"
]

class ChatbotUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests

    @task
    def send_rag_query(self):
        """Send a query to the RL-RAG API gateway."""
        query = SAMPLE_QUERIES[hash(self) % len(SAMPLE_QUERIES)]
        self.client.post(
            "/query",
            json={"query": query},
            headers={"Content-Type": "application/json"},
            name="/query"
        )
