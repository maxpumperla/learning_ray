#!/usr/bin/env python
# tag::server_config[]
# policy_server.py
import ray
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env.policy_server_input import PolicyServerInput
import gym


ray.init()


def policy_input(context):
    return PolicyServerInput(context, "localhost", 9900)  # <1>


config = {
    "env": None,  # <2>
    "observation_space": gym.spaces.Discrete(5*5),
    "action_space": gym.spaces.Discrete(4),
    "input": policy_input,  # <3>
    "num_workers": 0,
    "input_evaluation": [],
    "log_level": "INFO",
}

trainer = DQNTrainer(config=config)

# end::server_config[]

# tag::server_run[]
# policy_server.py
if __name__ == "__main__":

    time_steps = 0
    for _ in range(100):
        results = trainer.train()
        checkpoint = trainer.save()  # <1>
        if time_steps >= 10.000:  # <2>
            break
        time_steps += results["timesteps_total"]
# end::server_run[]
