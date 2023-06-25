#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 13:24:55 2023

@author: franciscorealescastro
"""

from agents.t_077.myTeam import myAgent as randomAgent

#from agents.generic.random import myAgent as randomAgent


from VectorState import toVectorState,vectorizeAction,descreteAction,descreteToMultihot,selectActionFromInteger

import importlib
import random,time, copy
import gym
from gym import spaces
import numpy as np
from   func_timeout import func_timeout, FunctionTimedOut
from   template     import Agent as DummyAgent

from copy import deepcopy
  
class AzulEnv(gym.Env):
    metadata = {'render.modes': ['console']}
  
    def __init__(self):
        super(AzulEnv, self).__init__()

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        
        # Example for using image as input:
        self.stateShape=159
        self.actionShape=150
        self.action_space = spaces.Discrete(self.actionShape)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.stateShape,), dtype=np.float64)
        self.opponentId=0
        self.opponent =randomAgent(self.opponentId)#opponent id=0, player id=1
        self.gamemaster = DummyAgent(2)
        self.endOfGameR=100
        
        game_name="Azul"
        self.model = importlib.import_module(f"{game_name}.{game_name.lower()}_model")
        GameRule = getattr(self.model, f'{game_name}GameRule')
        self.game_rule=GameRule(2)
        # Initialize state
        self.maxRewardBonus=0
        self.state = np.zeros(self.stateShape)
        
        
    def immediateReward(self,game_state,agent_index,selected):
        
        auxiliarAgentState=deepcopy(game_state.agents[agent_index])
        
        r_grid_floor=auxiliarAgentState.ScoreRound()[0]
        
        r_bonus=max(auxiliarAgentState.EndOfGameScore(),self.maxRewardBonus)-self.maxRewardBonus#bonus for sets+columns+rows (only delta is taken)
        self.maxRewardBonus=r_bonus
        
        r_well_potitioned=-1 if selected[2].num_to_floor_line>0 else 0.5
        
        return r_grid_floor+r_bonus+r_well_potitioned
    
  
    def step(self, action):
        # Execute one time step within the environment
        # Here you should modify the state given the action and return the new state, reward, done and info
        # print("action stable_baselines",action)
        
        
        agent_0_played = False
        agent_1_played = False
        selectedActions=2*[None]
        reward=0
        while True:
            agent_index = self.game_rule.getCurrentAgentIndex()#0,1 or 2
            game_state = self.game_rule.current_game_state
            game_state.agent_to_move = agent_index
            
            actions = self.game_rule.getLegalActions(game_state, agent_index)
            actions_copy = copy.deepcopy(actions)
            gs_copy = copy.deepcopy(game_state)
            
            if agent_index==self.opponentId:#opponent
                selected=self.selectActionOpponent(self.opponent,actions_copy,gs_copy)
                selectedActions[self.opponentId]=selected
            elif agent_index==1-self.opponentId:#player
                selected=selectActionFromInteger(action,actions_copy,mode="similarity")
                selectedActions[1-self.opponentId]=selected
                #selected=self.selectActionOpponent(self.opponent,actions_copy,gs_copy)
            elif agent_index==2:#game master
                selected=self.selectActionOpponent(self.gamemaster,actions_copy,gs_copy)
                
            
            random.seed(random.randint(0,1e10))
            if self.game_rule.getCurrentAgentIndex()==0:
                agent_0_played=True
            if self.game_rule.getCurrentAgentIndex()==1:
                agent_1_played=True
            # print("agent_0_played:",agent_0_played)
            # print("agent_1_played:",agent_1_played)
            #updates
            #self.game_rule.current_game_state and self.game_rule.getCurrentAgentIndex()  
            self.game_rule.update(selected)
            random.seed(random.randint(0,1e10))
            
            
            # print("agent_index: ",agent_index)
            # print("\n action selected: ",selected)
            
            # if agent_index<2:
            #     auxiliarAgentState=deepcopy(game_state.agents[agent_index])
            #     print("score agent",agent_index,"=",auxiliarAgentState.ScoreRound()[0])#score of the grid - floor line
            #     print("completed columns,rows,sets",agent_index,"=",auxiliarAgentState.EndOfGameScore())#sets+columns+rows (calculate delta)
            #     print("reward for well potitioned",agent_index,"=", -1 if selected[2].num_to_floor_line>0 else 0.5 )#0.5 if no tiles go to FL, -1 otherwise  
            
            
            if agent_0_played and agent_1_played and self.game_rule.getCurrentAgentIndex() in [0,1]:
                self.state=toVectorState(self.game_rule.current_game_state,1-self.opponentId)
                done=self.game_rule.gameEnds()
                info=0
                
                r_opponent=self.immediateReward(game_state,self.opponentId,selectedActions[self.opponentId])
                r_player=self.immediateReward(game_state,1-self.opponentId,selectedActions[1-self.opponentId])
                
                reward=r_player-r_opponent
                if done:
                    # Score agent bonuses
                    auxiliarAgentState=deepcopy(game_state.agents[self.opponentId])
                    scoreOpponent=auxiliarAgentState.EndOfGameScore()
                    
                    auxiliarAgentState=deepcopy(game_state.agents[1-self.opponentId])
                    scorePlayer=auxiliarAgentState.EndOfGameScore()
                    
                    if scorePlayer>scoreOpponent:
                        endOfGameReward= self.endOfGameR
                        info=1
                        #print("Player Won")
                    elif scorePlayer<=scoreOpponent:
                        endOfGameReward= - self.endOfGameR 
                        info=-1
                        #print("Opponent Won")
                    else: 
                        #endOfGameReward=0
                        info=0
                        #print("Tie")
                    reward+= endOfGameReward
                reward=(reward+self.endOfGameR)/(2*self.endOfGameR) #normalized reward but for large self.endOfGameR tends to be 0.5
                reward=2*(reward-0.5)#reward between -1 and 1 (almost)
                return self.state, reward, done, info
  
    def reset(self):
        # Reset the state of the environment to an initial state
        
        self.opponent =randomAgent(self.opponentId)#opponent can be updated when episode finishes

        
        game_name="Azul"
        self.model = importlib.import_module(f"{game_name}.{game_name.lower()}_model")
        GameRule = getattr(self.model, f'{game_name}GameRule')
        self.game_rule=GameRule(2)
        # Initialize state
        self.state = np.zeros(self.stateShape)
        random.seed(int(str(time.time()).replace('.', '')))
        self.maxRewardBonus=0
        return self.state
  
    def selectActionOpponent(self,agent,actions,state):
        valid_action = self.game_rule.validAction
        selected=random.choice(actions)
        try: 
            selected = func_timeout(1,agent.SelectAction,args=(actions,state))
        except FunctionTimedOut:
            selected = "timeout"
        except Exception as e:
            exception = e
            
        if self.game_rule.getCurrentAgentIndex() != self.game_rule.num_of_agent:
            if selected != "timeout":
                if valid_action:
                    if not valid_action(selected, actions):
                        selected = "illegal"
                elif not selected in actions:
                    selected = "illegal"
                
            if selected in ["timeout", "illegal"]:
                selected = random.choice(actions)
                
        return selected



# game_name="Azul"
# model = importlib.import_module(f"{game_name}.{game_name.lower()}_model")
# GameRule = getattr(model, f'{game_name}GameRule')

# random.seed(int(str(time.time()).replace('.', '')))

# #rewrite seed as random.randint(0,1e10)
# game_rule=GameRule(2)


# counter=0
# opponent =randomAgent(0)#opponent id=0, player id=1
# gamemaster = DummyAgent(2)

# stepDone=False
# agent_0_played = False
# agent_1_played = False


# while True:
#     agent_index = game_rule.getCurrentAgentIndex()#0,1 or 2
#     game_state = game_rule.current_game_state
#     game_state.agent_to_move = agent_index
    
#     actions = game_rule.getLegalActions(game_state, agent_index)
#     actions_copy = copy.deepcopy(actions)
#     gs_copy = copy.deepcopy(game_state)
#     print(agent_index)
#     if agent_index==1:#opponent
#         selected=selectActionOpponent(opponent,actions_copy,gs_copy)
#     elif agent_index==0:#player
#         #selected=action passed in step(action) but transformed to game object or take closest
#         selected=selectActionOpponent(opponent,actions_copy,gs_copy)
#     elif agent_index==2:#game master
#         selected=selectActionOpponent(gamemaster,actions_copy,gs_copy)
        
#     random.seed(random.randint(0,1e10))
#     if game_rule.getCurrentAgentIndex()==0:
#         agent_0_played=True
#     if game_rule.getCurrentAgentIndex()==1:
#         agent_1_played=True
#     print("agent_0_played:",agent_0_played)
#     print("agent_1_played:",agent_1_played)
#     #updates
#     #self.game_rule.current_game_state and self.game_rule.getCurrentAgentIndex()  
#     game_rule.update(selected)
#     random.seed(random.randint(0,1e10))
    
#     print("agent_index: ",agent_index)
#     print("\n action selected: ",selected)
#     print("pasoooo")
#     if agent_0_played and agent_1_played and game_rule.getCurrentAgentIndex() in [0,1]:

#         break
        
    