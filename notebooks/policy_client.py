#!/usr/bin/env python
# tag::client[]
# policy_client.py
import gym
from ray.rllib.env.policy_client import PolicyClient
from maze_gym_env import GymEnvironment

if __name__ == "__main__":
    env = GymEnvironment()
    client = PolicyClient("http://localhost:9900", inference_mode="remote")  # <1>

    obs = env.reset()
    episode_id = client.start_episode(training_enabled=True)  # <2>

    while True:
        action = client.get_action(episode_id, obs)  # <3>

        obs, reward, done, info = env.step(action)

        client.log_returns(episode_id, reward, info=info)  # <4>

        if done:
            client.end_episode(episode_id, obs)  # <5>
            obs = env.reset()

            exit(0)  # <6>

# end::client[]
