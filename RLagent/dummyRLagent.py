#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:24:55 2023

@author: franciscorealescastro
"""

"""""
from gymAzulModel import AzulEnv


from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

logDir='/home/francisco/Documents/aiplanning/assignment3-azul--lincoln_crew-master/RLagent/logs'
checkpointPath = "/home/francisco/Documents/aiplanning/assignment3-azul--lincoln_crew-master/RLagent/checkpoint"


checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=checkpointPath, name_prefix='checkpoint')


env=AzulEnv()
net_arch = [150, 150,150] 
policy_kwargs = dict(net_arch=net_arch)
model = PPO("MlpPolicy", env, verbose=1,tensorboard_log=logDir,policy_kwargs=policy_kwargs)
print(model.policy)
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("USing device: ",device)
model.learn(total_timesteps=100000000,callback=checkpoint_callback)

"""

from gymAzulModel import AzulEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

logDir='/home/francisco/Documents/aiplanning/assignment3-azul--lincoln_crew-master/RLagent/logs'
checkpointPath = "/home/francisco/Documents/aiplanning/assignment3-azul--lincoln_crew-master/RLagent/checkpoint"

checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=checkpointPath, name_prefix='checkpoint')

env=AzulEnv()
num_envs = 8  # Set this to the number of environments you want to run in parallel.
env = DummyVecEnv([lambda: AzulEnv() for _ in range(num_envs)])


net_arch = [150, 150,150] 
policy_kwargs = dict(net_arch=net_arch)

model = PPO("MlpPolicy", env, verbose=1,tensorboard_log=logDir,policy_kwargs=policy_kwargs)
#model.load('/home/francisco/Documents/aiplanning/assignment3-azul--lincoln_crew-master/RLagent/modelckpt/checkpoint_6800000_steps.zip')

print(model.policy)
import torch 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ",device)

from stable_baselines3.common.callbacks import EvalCallback

eval_env = AzulEnv()
eval_callback = EvalCallback(eval_env, best_model_save_path=logDir,
                             log_path=logDir, eval_freq=100000,
                             deterministic=True, render=False)

model.learn(total_timesteps=100000000,callback=[checkpoint_callback, eval_callback])
