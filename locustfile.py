from locust import HttpUser, task, between

class PredictUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def predict_normal_log(self):
        self.client.post(
            "/predict",
            json={"log_text": "Technician report: Routine check completed on gearbox. No issues found."}
        )