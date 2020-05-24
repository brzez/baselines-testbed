import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

import pybullet_envs

env = gym.make('MinitaurBulletEnv-v0', render=True)

# multiprocess environment
# env = make_vec_env('MinitaurBulletEnv-v0', n_envs=4)
model = PPO2.load("ppo2")

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()