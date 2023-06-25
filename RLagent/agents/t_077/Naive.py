# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Paul Kretzschmar
# Date:    03/05/2023
# Purpose: Implements an minimax search agent for the COMP90054 competitive game environment.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import time, random
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque
import math
import numpy as np

THINKTIME   = 0.9
NUM_PLAYERS = 2


# FUNCTIONS ----------------------------------------------------------------------------------------------------------#


class myAgent():
    def __init__(self, _id):
        self.id = _id # Agent needs to remember its own id.
        self.game_rule = GameRule(NUM_PLAYERS) # Agent stores an instance of GameRule, from which to obtain functions.
        # More advanced agents might find it useful to not be bound by the functions in GameRule, instead executing
        # their own custom functions under GetActions and DoAction.


    def get_remainder(self, player_state, line_idx, num_to_line):
        remainder = 0

        if player_state.lines_tile[line_idx] != -1:
            num_exist = player_state.lines_number[line_idx]
            remainder = line_idx + 1 - (num_exist + num_to_line)

        else:
            assert player_state.lines_number[line_idx] == 0
            remainder = line_idx + 1 - num_to_line

        return remainder    

    def SelectAction(self, actions, state):
        # Select move that involves placing the most number of tiles
        # in a pattern line. Tie break on number placed in floor line.
        most_to_line = -1
        corr_to_floor = 0
        best_remainder = 5

        best_move = None
        #print(f"action: {actions[0]}")
        #line_tile = state.agents[self.id].lines_tile
        #line_number = state.agents[self.id].lines_number
        
        for action_id,factory_id,tile_grab in actions:

            num_to_line = tile_grab.num_to_pattern_line
            line_dest = tile_grab.pattern_line_dest
            #remainder = self.get_remainder(state.agents[self.id], line_dest, num_to_line)
            
            if most_to_line == -1:
                best_move = (action_id,factory_id,tile_grab)
                most_to_line= num_to_line
                corr_to_floor = tile_grab.num_to_floor_line
                best_remainder = self.get_remainder(state.agents[self.id], line_dest, num_to_line)
                continue

            if tile_grab.num_to_pattern_line > most_to_line:
                best_move = (action_id,factory_id,tile_grab)
                most_to_line = num_to_line
                corr_to_floor = tile_grab.num_to_floor_line
                best_remainder = self.get_remainder(state.agents[self.id], line_dest, num_to_line)
            
            elif tile_grab.num_to_pattern_line == most_to_line and \
                tile_grab.num_to_floor_line < corr_to_floor:
                best_move = (action_id,factory_id,tile_grab)
                most_to_line = num_to_line
                corr_to_floor = tile_grab.num_to_floor_line
                best_remainder = self.get_remainder(state.agents[self.id], line_dest, num_to_line)
            
            elif tile_grab.num_to_pattern_line == most_to_line and \
                tile_grab.num_to_floor_line == corr_to_floor and\
                self.get_remainder(state.agents[self.id], line_dest, num_to_line) < best_remainder:
                best_move = (action_id,factory_id,tile_grab)
                most_to_line = num_to_line
                corr_to_floor = tile_grab.num_to_floor_line    
                best_remainder = self.get_remainder(state.agents[self.id], line_dest, num_to_line)
        return best_move