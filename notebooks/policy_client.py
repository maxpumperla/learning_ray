#!/usr/bin/env python
import gym
from ray.rllib.env.policy_client import PolicyClient
from maze_gym_env import GymEnvironment

if __name__ == "__main__":
    env = GymEnvironment()
    client = PolicyClient("http://localhost:9900", inference_mode="remote")

    obs = env.reset()
    episode_id = client.start_episode(training_enabled=True)

    while True:
        action = client.get_action(episode_id, obs)

        obs, reward, done, info = env.step(action)

        client.log_returns(episode_id, reward, info=info)

        if done:
            client.end_episode(episode_id, obs)
            exit(0)
