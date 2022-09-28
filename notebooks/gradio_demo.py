import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# TODO it complains about missing "model_definition", but only "model" exists.
# Also, we need to make clear that users know that "model" is required here and which
# format it expects.
# TODO note that one can provide preprocessors as well


# tag::load_checkpoint[]
from ray.train.torch import TorchCheckpoint, TorchPredictor

CHECKPOINT_PATH = "torch_checkpoint"
checkpoint = TorchCheckpoint.from_directory(CHECKPOINT_PATH)
predictor = TorchPredictor.from_checkpoint(
    checkpoint=checkpoint,
    model=Net()
)
# end::load_checkpoint[]

# tag::gradio[]
from ray.serve.gradio_integrations import GradioServer
import gradio as gr
import numpy as np


def predict(payload):  # <1>
    payload = np.array(payload, dtype=np.float32)
    array = payload.reshape((1, 3, 32, 32))
    return np.argmax(predictor.predict(array))


demo = gr.Interface(  # <2>
    fn=predict,
    inputs=gr.Image(),
    outputs=gr.Label(num_top_classes=10)
)

app = GradioServer.options(  # <3>
    num_replicas=2,
    ray_actor_options={"num_cpus": 2}
).bind(demo)
# end::gradio[]

# demo.launch()


# TODO none of this ever properly fails if incorrectly configured
# import ray
# from ray import serve
# from ray.serve.gradio_integrations import GradioIngress
#
# import gradio as gr
#
#
# @serve.deployment
# class TorchClassifier:
#     def __call__(self, data):
#         return predict(data)
#
# app = TorchClassifier.bind()
#
#
# @serve.deployment
# class MyGradioServer(GradioIngress):
#     def __init__(self, deployment_1, deployment_2):
#         self.d1 = deployment_1
#         self.d2 = deployment_2
#
#         io = gr.Interface(
#             fn=self.fan_out,
#             inputs=gr.Image(),
#             outputs="textbox",
#         )
#         super().__init__(io)
#
#     def fan_out(self, array):
#         [result0, result1] = ray.get([self.d1.remote(array), self.d2.remote(array)])
#         return f"First model: {result0}, second model: {result1}"
#
#
# app = MyGradioServer.bind(app, app)
