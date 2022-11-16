#!/usr/bin/env python
# tag::server_config[]
# policy_server.py
import ray
from ray.rllib.agents.dqn import DQNConfig
from ray.rllib.env.policy_server_input import PolicyServerInput
import gym


ray.init()


def policy_input(context):
    return PolicyServerInput(context, "localhost", 9900)  # <1>


config = DQNConfig()\
    .environment(
        env=None,  # <2>
        action_space=gym.spaces.Discrete(4),  # <3>
        observation_space=gym.spaces.Discrete(5*5))\
    .debugging(log_level="INFO")\
    .rollouts(num_rollout_workers=0)\
    .offline_data(  # <4>
        input=policy_input,
        input_evaluation=[])\


algo = config.build()

# end::server_config[]

# tag::server_run[]
# policy_server.py
if __name__ == "__main__":

    time_steps = 0
    for _ in range(100):
        results = algo.train()
        checkpoint = algo.save()  # <1>
        if time_steps >= 1000:  # <2>
            break
        time_steps += results["timesteps_total"]
# end::server_run[]
