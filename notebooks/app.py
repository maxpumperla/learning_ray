# In this file we collect all the Serve deployments that get referenced in the chapter.
# You can run any of these deployments by running `serve run app:<deployment_name>`,
# where <deployment_name> is any of basic_deployment, scaled_deployment,
# nlp_pipeline_driver, or batched_deployment.

from fastapi import FastAPI
from transformers import pipeline
from ray import serve


app = FastAPI()


@serve.deployment
class SentimentAnalysis:
    def __init__(self):
        self._classifier = pipeline("sentiment-analysis")

    def __call__(self, request) -> str:
        input_text = request.query_params["input_text"]
        return self._classifier(input_text)[0]["label"]


basic_deployment = SentimentAnalysis.bind()


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 2})
@serve.ingress(app)
class SentimentAnalysis:
    def __init__(self):
        self._classifier = pipeline("sentiment-analysis")

    @app.get("/")
    def classify(self, input_text: str) -> str:
        import os
        print("from process:", os.getpid())
        return self._classifier(input_text)[0]["label"]


scaled_deployment = SentimentAnalysis.bind()


@serve.deployment
@serve.ingress(app)
class SentimentAnalysis:
    def __init__(self):
        self._classifier = pipeline("sentiment-analysis")

    @serve.batch(max_batch_size=10, batch_wait_timeout_s=0.1)
    async def classify_batched(self, batched_inputs):
        print("Got batch size:", len(batched_inputs))
        results = self._classifier(batched_inputs)
        return [result["label"] for result in results]

    @app.get("/")
    async def classify(self, input_text: str) -> str:
        return await self.classify_batched(input_text)


batched_deployment = SentimentAnalysis.bind()
