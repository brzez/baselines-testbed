import gym
import numpy as np

import tensorflow as tf
import pybullet_envs
from datetime import datetime

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

last_time = datetime.now().timestamp()

net_name = "ppo2_2x64"
policy_kwargs = dict(
    net_arch=[64, 64]
)

def cb(a, b):
    global last_time
    t = datetime.now().timestamp()
    if t - last_time > 60:
        last_time = t
        print("SAVING===" * 10)
        model.save(net_name)

# multiprocess environment
env = make_vec_env('MinitaurBulletEnv-v0', n_envs=4)
# env = gym.make('MinitaurBulletEnv-v0', render=True)

try:
    model = PPO2.load(
        net_name,
        policy_kwargs=policy_kwargs,
        env=env
    )
except ValueError:
    model = PPO2(
        MlpPolicy, 
        env, 
        verbose=1,
        tensorboard_log='./tensorboard',
    )

while True:
    model.learn(
        total_timesteps=2000000,
        callback=cb,
        tb_log_name=net_name
    )
    model.save(net_name)
