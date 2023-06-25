# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Steven Spratley
# Date:    04/01/2021
# Purpose: Implements an example breadth-first search agent for the COMP90054 competitive game environment.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import time, random
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque
import numpy as np
THINKTIME   = 0.9
NUM_PLAYERS = 2


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#


# Defines this agent.
class myAgent():
    def __init__(self, _id):
        self.id = _id # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS) # Agent stores an instance of GameRule, from which to obtain functions.
        # More advanced agents might find it useful to not be bound by the functions in GameRule, instead executing
        # their own custom functions under GetActions and DoAction.

    # Generates actions from this state.
    def GetActions(self, state):
        return self.game_rule.getLegalActions(state, self.id)

        

    # Carry out a given action on this state and return True if goal is reached received.
    def DoAction(self, state, action):
        score = state.agents[self.id].score
        #print("Score: ",score)
        
        score1=np.sum(state.agents[self.id].lines_number)
        state = self.game_rule.generateSuccessor(state, action, self.id)
        score2=np.sum(state.agents[self.id].lines_number)
        
        if score2<=score1: print("s2>=s1")
        goal_reached = False #TODO: Students, how should agent check whether it reached goal or not
        
        if score2>score1: 
            print("s2>=s1 goal reached")
            goal_reached=True
            
        
        # goal_reached=False
        # if  not isinstance(action,str) and state.agents[self.id].lines_number[action[2].pattern_line_dest] == state.agents[self.id].GRID_SIZE:
        #     goal_reached=True
        #     print("Line in pattern Line filled: ",action[2].pattern_line_dest)
            
        
        return goal_reached

    # Take a list of actions and an initial state, and perform breadth-first search within a time limit.
    # Return the first action that leads to goal, if any was found.
    def SelectAction(self, actions, state):
        start_time = time.time()
        #queue      = deque([ (deepcopy(rootstate),[]) ]) # Initialise queue. First node = root state and an empty path.
        
        # Conduct BFS starting from rootstate.
        #while len(queue) and time.time()-start_time < THINKTIME:
            #state, path = queue.popleft() # Pop the next node (state, path) in the queue.
        new_actions = self.GetActions(state) # Obtain new actions available to the agent in this state.
        

        if len(new_actions)==1: return new_actions[0]
        bestAction=[None,np.inf]
        for a in new_actions: # Then, for each of these actions...
            
            
            if  not isinstance(a,str) :
                tg=a[2]
                freeSlots=((tg.pattern_line_dest+1) - state.agents[self.id].lines_number[tg.pattern_line_dest]  )
                if tg.num_to_floor_line==0:
                    bestFunction=freeSlots
                else:
                    bestFunction=tg.num_to_floor_line
                
                if bestFunction<bestAction[1]:
                    bestAction[1]=bestFunction
                    bestAction[0]=a
        return bestAction[0]
                    
                
            
# END FILE -----------------------------------------------------------------------------------------------------------#