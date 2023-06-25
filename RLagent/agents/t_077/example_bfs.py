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
    def SelectAction(self, actions, rootstate):
        start_time = time.time()
        queue      = deque([ (deepcopy(rootstate),[]) ]) # Initialise queue. First node = root state and an empty path.
        
        # Conduct BFS starting from rootstate.
        while len(queue) and time.time()-start_time < THINKTIME:
            state, path = queue.popleft() # Pop the next node (state, path) in the queue.
            new_actions = self.GetActions(state) # Obtain new actions available to the agent in this state.
            
            for a in new_actions: # Then, for each of these actions...
                next_state = deepcopy(state)              # Copy the state.
                next_path  = path + [a]                   # Add this action to the path.
                goal     = self.DoAction(next_state, a) # Carry out this action on the state, and check for goal
                if goal:
                    #print(f'Move {self.turn_count}, path found:', next_path)
                    return next_path[0] # If the current action reached the goal, return the initial action that led there.
                else:
                    queue.append((next_state, next_path)) # Else, simply add this state and its path to the queue.
        print("executing random action")
        return random.choice(actions) # If no goal was found in the time limit, return a random action.
        
    
# END FILE -----------------------------------------------------------------------------------------------------------#