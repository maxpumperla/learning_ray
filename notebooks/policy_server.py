#!/usr/bin/env python
import ray
from ray.rllib.agents.dqn import DQNConfig
from ray.rllib.env.policy_server_input import PolicyServerInput
import gym


ray.init()


def policy_input(context):
    return PolicyServerInput(context, "localhost", 9900)


config = DQNConfig()\
    .environment(
        env=None,
        action_space=gym.spaces.Discrete(4),
        observation_space=gym.spaces.Discrete(5*5))\
    .debugging(log_level="INFO")\
    .rollouts(num_rollout_workers=0)\
    .offline_data(
        input=policy_input,
        input_evaluation=[])\


algo = config.build()


if __name__ == "__main__":

    time_steps = 0
    for _ in range(100):
        results = algo.train()
        checkpoint = algo.save()
        if time_steps >= 1000:
            break
        time_steps += results["timesteps_total"]
