# locustfile.py
from locust import HttpUser, task, between
import random
import json

class BackendUser(HttpUser):
    wait_time = between(0.5, 2)

    @task(5)
    def simple_query(self):
        payload = {"text": "Benchmark query example number " + str(random.randint(1,100000))}
        headers = {"Content-Type": "application/json"}
        self.client.post("/api/query", data=json.dumps(payload), headers=headers, name="/api/query")

    @task(1)
    def multipart_example(self):
        # if you want to test the multipart endpoint, Locust can do that too
        files = {"upload_file": ("dummy.txt", "hello world")}
        self.client.post("/api/query-multipart", files=files, name="/api/query-multipart")
