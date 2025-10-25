from locust import HttpUser, task, between

class ChatbotUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def send_message(self):
        self.client.get("/")  # replace with actual endpoint later
