from stable_baselines3 import PPO
import numpy as np
from gymAzulModel import AzulEnv
# Load the trained model from a checkpoint
import glob
from tqdm import tqdm

path="/home/francisco/Documents/aiplanning/assignment3-azul--lincoln_crew-master/RLagent/checkpointPred/"
trainingSteps=[]
winnings=[]
ties=[]
loses=[]
for file in tqdm(glob.glob(path+'*.zip')):
    trainingSteps.append(file.split("_")[2])

    trained_model = PPO.load(file)

    # Create environment
    env = AzulEnv()
    #info_games=[]
    # Evaluate the policy by using deterministic actions
    obs = env.reset()
    i=0
    countWin=0
    countLose=0
    countTie=0
    while i<100:
        action, _states = trained_model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if done:
            i+=1
            #info_games.append(info)
            obs = env.reset()
            if info==1:
                countWin+=1
            if info==-1:
                countLose+=1
            if info==0:
                countTie+=1
    winnings.append(countWin)
    ties.append(countTie)
    loses.append(countLose)

import matplotlib.pyplot as plt

# Convert trainingSteps to int
trainingSteps = [int(step) for step in trainingSteps]

# Combine the lists and sort by trainingSteps
combined = sorted(zip(trainingSteps, winnings, ties, loses))

# Unzip the sorted combined list back to individual lists
trainingSteps, winnings, ties, loses = zip(*combined)

plt.figure(figsize=(10, 6))

plt.plot(trainingSteps, winnings, label='Wins', marker='o')
plt.plot(trainingSteps, ties, label='Ties', marker='o')
plt.plot(trainingSteps, loses, label='Losses', marker='o')

plt.xlabel('Training Steps', fontsize=14)
plt.ylabel('Number of Games (100 were played)', fontsize=14)
plt.title('Performance of PPO Agent Against MCTS', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.tight_layout()
plt.show()
