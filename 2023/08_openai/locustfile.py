from locust import HttpUser, task

from mock_openai.models import Mode


class User(HttpUser):
    @task
    def make_request(self):
        for mode in Mode:
            self.client.get("/test-proxy", params={"mode": mode.value})
