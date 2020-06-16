import gym
from modelbased import ModelAgent
import random
import numpy as np
import torch
from gym import wrappers


def scale_action(env, action):
    return (env.action_space.high-env.action_space.low)*action + env.action_space.low


def run_episode(env, agent):
    obs = env.reset()
    reward_tot = 0.0
    done = False
    reward = 0.0
    while not done:
        action = scale_action(env, agent.step(obs, reward))
        obs, reward, done, _ = env.step(action)
        reward_tot += reward
        env.render()
    agent.step(obs, reward)
    agent.reset()
    return reward_tot

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    env_to_wrap = gym.make("Pendulum-v0")
    env = wrappers.Monitor(env_to_wrap, 'logging/', force=True, video_callable=lambda episode_id: True)

    agent = ModelAgent(env.observation_space.shape[0], env.action_space.shape[0])

    for i in range(10):
       reward_tot = run_episode(env, agent)
       print("Episode: ", i+1, "---", "Total Reward: ", reward_tot)
    env.close()
