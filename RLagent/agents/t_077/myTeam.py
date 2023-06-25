# INFORMATION ------------------------------------------------------------------------------------------------------- #


# Author:  Simone Marchetti
# Date:    22/05/2023
# Purpose: Implement MCTS for the COMP90054 competitive game environment.


# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


import time, random
from Azul.azul_model import AzulGameRule as GameRule
from copy import deepcopy
from collections import deque
import numpy as np
import math

THINKTIME   = 0.9
NUM_PLAYERS = 2

# FUNCTIONS ----------------------------------------------------------------------------------------------------------#

dict_q_values = {}
global_var = []
#DEFINE MCT
class Node(object):
    def __init__(self, parent, game_state, id, current_action):
        self.id = id
        #self.q_value = dict_q_values.get((game_state, current_action), 0)
        #self.q_value = dict_q_values.get(game_state, 0)
        self.q_value = 0
        self.parent = parent
        self.number_visited = 0
        self.game_state = game_state
        self.children = list() #list MCT node
        self.children_q_values = list()
        self.children_uct = list()
        self.children_queue = PriorityQueue()
        #self.game_rule = game_rule
        self.game_rule = GameRule(NUM_PLAYERS)
        self.legal_actions =self.game_rule.getLegalActions(game_state, self.id)
        self.current_action = current_action
        self.rewards_ = [0,0]
        #self.action_queue = PriorityQueue()
        #self.player_id = id

    def is_fully_expanded(self):
        #if any([child.number_visited != 0 for child in self.children]):
        #    return False
        #for child in self.children:
        #    if child.visit_times == 0:
        #        return False
        for child in self.children:
            if child.number_visited == 0:
                return False
        return True
        #return True

    def get_new_child(self):
        for child in self.children:
            if child.number_visited == 0:
                return child
        return None

    def calculate_q(self):
        if self.number_visited == 0:
            return 0
        ##print(f"father wins: {self.wins[1-self.id]}")
        return self.rewards_[1-self.id]/self.number_visited

    def get_best_q_value_child(self):
        #print("LOOKING FOR BEST CHILDREN")
        #best_child = self.children[0]
        #for child in self.children:
        #    if child.q_value > best_child.q_value:
        #        best_child = child
        #return self.children[np.argmax(self.children_q_values)]
        #return best_child
        best_child = self.children[0]
        for child in self.children:
            #print("inside best")
            if child.q_value> best_child.q_value:
                ##print("ENTER IN COMPARISON")
                best_child = child
        return best_child

    def get_best_uct_child(self):
        #print("#########BEST UCT CHILD#######")
        best_child = self.children[0]
        max_q, min_q = 0,0
        ##print(children_q_values)
        #max_q, min_q = max(children_q_values), min(children_q_values)
        normalize = False
        for child in self.children:
            if child.calculate_uct(max_q, min_q, False) > best_child.calculate_uct(max_q, min_q,False):
                best_child = child
        ##print(f"best child: {best_child.game_state}")
        return best_child


    def calculate_uct(self,max_q, min_q, normalize):
        if self.number_visited == 0:
            return float("inf")
        exploration_constant = 1
        if normalize==False:
            return self.q_value + 2*exploration_constant * math.sqrt((math.log(self.parent.number_visited)) / self.number_visited)
        uct = (self.q_value -min_q)/(max_q - min_q) + 2*exploration_constant * math.sqrt((2*math.log(self.parent.number_visited)) / self.number_visited)
        return uct

    def update(self, G, next):
        if next is None:
            self.q_value = self.rewards_
            return G
        ##print("UPDATING Q-VALUES")
        discount = 0.9
        p = self.game_state.agents[self.id]
        p2 = self.game_state.agents[1-self.id]
        """parent = self.parent
        if parent is None:
            current_reward = p.ScoreRound()[0]
        else:
             current_reward = p.ScoreRound()[0] -self.parent.game_state.agents[self.id].ScoreRound()[0]
        ##print(np.sum(p.lines_number))"""
        #current_reward = 
        current_reward = np.sum(p.lines_number)
        self.number_visited += 1
        #print(G)
        G[self.id] = current_reward + discount*G[self.id]
        #print(G)
        #print("before error")
        self.q_value[self.id] += (1/self.number_visited)*(G[self.id]-self.q_value[self.id])
        #print("after error")
        self.q_value[1-self.id] =  next.q_value[1-self.id]
        #children_q_values = [child.q_value for child in self.children]
        #G[1] = current_reward[1] + discount*G[1]
        ##print(G)
        #self.q_value = self.q_value + (1/self.number_visited)*(G-self.q_value)
        #self.wins += G[self.id]
        #self.q_value = self.wins / self.number_visited
        ##print(f"q: {self.q_value}")
        #self.q_value = s1 / s2
        #dict_q_values[self.game_state] = self.q_value
        ##print(self.q_value)
        #r = self.game_rule
        #self.q_value += 
        ##print(self.game_state)
        ##print(self.q_value)
        #print("returning G")
        return G

    def is_round_end(self):
        return not self.game_state.TilesRemaining()



def SelectNaiveAction(actions):
        # Select move that involves placing the most number of tiles
        # in a pattern line. Tie break on number placed in floor line.
        most_to_line = -1
        corr_to_floor = 0

        best_move = None
        ##print(f"action: {actions[0]}")

        for action_id,factory_id,tile_grap in actions:
            if most_to_line == -1:
                best_move = (action_id,factory_id,tile_grap)
                most_to_line = tile_grap.num_to_pattern_line
                corr_to_floor = tile_grap.num_to_floor_line
                continue

            if tile_grap.num_to_pattern_line > most_to_line:
                best_move = (action_id,factory_id,tile_grap)
                most_to_line = tile_grap.num_to_pattern_line
                corr_to_floor = tile_grap.num_to_floor_line
            elif tile_grap.num_to_pattern_line == most_to_line and \
                tile_grap.num_to_pattern_line < corr_to_floor:
                best_move = (action_id,factory_id,tile_grap)
                most_to_line = tile_grap.num_to_pattern_line
                corr_to_floor = tile_grap.num_to_floor_line

        return best_move

def get_remainder(player_state, line_idx, num_to_line):
        remainder = 0

        if player_state.lines_tile[line_idx] != -1:
            num_exist = player_state.lines_number[line_idx]
            remainder = line_idx + 1 - (num_exist + num_to_line)

        else:
            assert player_state.lines_number[line_idx] == 0
            remainder = line_idx + 1 - num_to_line

        return remainder    

def SelectNaiveAction2(actions, state, id):
        # Select move that involves placing the most number of tiles
        # in a pattern line. Tie break on number placed in floor line.
        most_to_line = -1
        corr_to_floor = 0
        best_remainder = 5

        best_move = None
        ##print(f"action: {actions[0]}")
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
                best_remainder = get_remainder(state.agents[id], line_dest, num_to_line)
                continue

            if tile_grab.num_to_pattern_line > most_to_line:
                best_move = (action_id,factory_id,tile_grab)
                most_to_line = num_to_line
                corr_to_floor = tile_grab.num_to_floor_line
                best_remainder = get_remainder(state.agents[id], line_dest, num_to_line)
            
            elif tile_grab.num_to_pattern_line == most_to_line and \
                tile_grab.num_to_floor_line < corr_to_floor:
                best_move = (action_id,factory_id,tile_grab)
                most_to_line = num_to_line
                corr_to_floor = tile_grab.num_to_floor_line
                best_remainder = get_remainder(state.agents[id], line_dest, num_to_line)
            
            elif tile_grab.num_to_pattern_line == most_to_line and \
                tile_grab.num_to_floor_line == corr_to_floor and\
                get_remainder(state.agents[id], line_dest, num_to_line) < best_remainder:
                best_move = (action_id,factory_id,tile_grab)
                most_to_line = num_to_line
                corr_to_floor = tile_grab.num_to_floor_line    
                best_remainder = get_remainder(state.agents[id], line_dest, num_to_line)
        return best_move

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
    def DoAction(self, state, action, id):
        return self.game_rule.generateSuccessor(state, action, id)

    # Take a list of actions and an initial state, and perform breadth-first search within a time limit.
    # Return the first action that leads to goal, if any was found.
    def SelectAction(self, actions, game_state):
        
        id = self.id
        best_move = None
        state = deepcopy(game_state)
        if len(actions) == 1:
            return actions[0]
        current_node = Node(None, state, id,  None)
        mcts = MCTS(current_node)
        best_child = mcts.search()
        best_move = best_child.current_action
        return best_move

def Action_simplification(moves):
    ans = []
    ##print("INSIDE SEMPLIFICATION")
    for move in moves:
        if len(moves) > 50:
            # Not allow picking one tile to fourth and fifth line at the beginning
            if move[2].pattern_line_dest > 2 and move[2].num_to_pattern_line == 1:
                continue
        #if move == "ENDROUND":
        #    continue
        if len(moves) > 30:
            # Not allow picking all the thing to the floor_line at the beginning
            if move[2].num_to_floor_line == move[2].number:
                continue
        #check = move[2].pattern_line_dest+1
        #if (check - move[2].num_to_pattern_line) < check/2:
        #    #print(check)
        #    #print("INSIDE CHECK")
        #    continue
        ans.append(move)
    #if len(ans)==0:
    #    return moves
    return ans


#DEFINE MCT

class MCTS(object):

    def __init__(self, root):
        self.root: Node = root

    def run_rollout(self):
        expand_node = self.Select(self.root)
        
        child = expand_node
        #print("before expansion")
        if expand_node.number_visited > 2 and expand_node.game_state.TilesRemaining():
            self.Expand(expand_node)
            if len(expand_node.children)>0:
                ##print("##########POPPING##########")
                #child = expand_node.children_queue.pop()
                child = random.choice(expand_node.children)
        #print("after expansion")
        
        reward = self.Simulation(child)
        
        self.Backpropagate(child, reward)

    def Select(self, current):
        while len(current.children) > 0:
            if current.number_visited == 0:
                return current
            current = current.get_best_uct_child()
        return current

    def Expand(self, expand_node):
        #print("INSIDE EXPAND")
        id = expand_node.id
        opponent_id = 1-expand_node.id
        actions = expand_node.legal_actions
        actions = Action_simplification(actions)
        #if len(actions) == 1:
        #    if actions[0] == "ENDROUND":
        #        return expand_node
        #selected_action = SelectNaiveAction(actions)
        #candidate = None
        for m in actions:
            game_state = deepcopy(expand_node.game_state)
            player = myAgent(expand_node.id)
            c = Node(parent=expand_node, game_state=player.DoAction(game_state, m,player.id), id=opponent_id, current_action=m)
            expand_node.children.append(c)
            #expand_node.children_queue.push(c, 1/np.sum(c.game_state.agents[player.id].lines_number))
            #if m == selected_action:
            #    candidate = c
        #if candidate is None:
        #    return random.choice(expand_node.children)
        #return candidate

    def Backpropagate(self, expand_node, reward):
        node = expand_node
        next = None
        discount_factor = 0.9
        discount_exponent = 0
        while node:
            """if node.parent == None:
                break"""
            #print("update")
            node.number_visited += 1
            """if next is None:
                immediate_reward = [0,0]
            else:"""
            if node.parent is None:
                immediate_reward = [0,0]
            else:
                immediate_reward = [np.sum(node.game_state.agents[0].lines_number) - np.sum(node.parent.game_state.agents[0].lines_number), \
                                    np.sum(node.game_state.agents[1].lines_number)- np.sum(node.parent.game_state.agents[1].lines_number)]

            node.rewards_[0] += immediate_reward[0] + reward[0]
            node.rewards_[1] += immediate_reward[1] + reward[1]
            node.q_value = node.calculate_q()
            next = node
            node = node.parent
        #print("####END OF BACKPROPAGATION")

    def Simulation(self, node: Node):

        state = deepcopy(node.game_state)
        ##print(f"starting count: {count}")
        ##print(f"tiles remaining: {state.TilesRemaining()}")
        move_simulation = 0
        current_player = node.id
        ##print("INSIDE SIMULATION")
        while state.TilesRemaining():

            ##print(state.agents[current_player])
            player = myAgent(current_player)
            actions =  player.GetActions(state)
            action = SelectNaiveAction2(actions, state, current_player)
            state = player.DoAction(state, action, player.id)
            move_simulation += 1
            current_player = 1 - current_player
        ##print("#########Out of loop#########")
        #state.ExecuteEndOfRound()
        #state.EndOfGameScore()
        reward = []
        ##print("####REWARD EVALUATION####")
        discount_factor = 0.9
        reward.append(state.agents[0].ScoreRound()[0] + reward_estimation(state, 0)) #* (discount_factor ** move_count)
        reward.append(state.agents[1].ScoreRound()[0]+ reward_estimation(state, 1))
        return [rew*(discount_factor**move_simulation) for rew in reward]

    def search(self):
        start_time = time.time()
        count = 1
        #print("NEW SEARCH")
        ##print(count)
        while time.time() - start_time < THINKTIME:
            #print("starting new")
            roll_out = self.run_rollout()
            count += 1
            #print(f"count: {count}")
            ##print(time.time()- start_time)
        #print(f"number of iterations {count}")
        ##print(f"numnber of children: {self.root.children}")
        ##print(f"out of loop value: {i}")
        #print("before error")
        child = self.root.get_best_q_value_child()
        #print(f"SELECTED CHILD ID: {child.id}")
        move = child.current_action
        #print(f"move search MCT: {move}")
        return child

# END FILE ----------

import heapq
class PriorityQueue:
    """
      Implements a priority queue data structure. Each inserted item
      has a priority associated with it and the client is usually interested
      in quick retrieval of the lowest-priority item in the queue. This
      data structure allows O(1) access to the lowest-priority item.
    """
    def  __init__(self):
        self.heap = []
        self.count = 0

    def getMinimumPriority(self):
        return self.heap[0][0]

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)






def col_row_bonus(game_state):
            used_tiles = []

            score_inc = 0
            state_copy = deepcopy(game_state)
            # 1. Action tiles across from pattern lines to the wall grid
            ##print("before error")
            ##print(game_state)
            for i in range(state_copy.GRID_SIZE):
                ##print("after error")
                # Is the pattern line full? If not it persists in its current
                # state into the next round.
                if state_copy.lines_number[i] == i+1:
                    tc = state_copy.lines_tile[i]
                    col = int(state_copy.grid_scheme[i][tc])

                    # Record that the agent has placed a tile of type 'tc'
                    state_copy.number_of[tc] += 1

                    # Clear the pattern line, add all but one tile into the
                    # used tiles bag. The last tile will be placed on the 
                    # agents wall grid.  
                    for j in range(i):
                        used_tiles.append(tc)

                    state_copy.lines_tile[i] = -1
                    state_copy.lines_number[i] = 0

                    # Tile will be placed at position (i,col) in grid
                    game_state.grid_state[i][col] = 1
            row_bonus = 0
            col_bonus = 0
            for i in range(state_copy.GRID_SIZE):
                row = np.sum(state_copy.grid_scheme[i][:]) - np.sum(game_state.grid_scheme[i][:])
                if row > 1:
                    row_bonus += row
                col = np.sum(state_copy.grid_scheme[:][i]) -  np.sum(game_state.grid_scheme[:][i])
                if col > 1:
                    col_bonus += col
            return row_bonus, col_bonus
                


def unfinished_pattern_lines(state):
    penalty = 0
    for i in range(1, 5):
        if state.lines_number[i] > 0:
            penalty += 1
    return penalty

def alone_tile_punishment(state):
    penalty = 0
    for i in range(5):
        above = 0
        for j in range(i-1, -1, -1):
            val = state.grid_state[i][j]
            above += val
            if val == 0:
                break
        below = 0
        for j in range(i+1,5,1):
            val = state.grid_state[i][j]
            below += val
            if val == 0:
                break
        left = 0
        for j in range(i-1, -1, -1):
            val = state.grid_state[i][j]
            left += val
            if val == 0:
                break
        right = 0
        for j in range(i+1, 5, 1):
            val = state.grid_state[i][j]
            right += val
            if val == 0:
                break
        if above == 0 and below == 0 and right == 0 and left == 0:
            penalty += 5
    return penalty

def penalty_evaluatipn(player_game_state):
    player_state = deepcopy(player_game_state)
    #player_state.ExecuteEndOfRound()
    penalty = 0
    # Punish unfinished pattern line
    penalty += unfinished_pattern_lines(player_state)
    # Extra Punishment for fourth pattern line
    if player_state.lines_number[3] == 1:
        penalty += 3
    if player_state.lines_number[3] == 2:
        penalty += 2
    if player_state.lines_number[3] == 3:
        penalty += 1
    # Extra Punishment for fifth pattern line
    if player_state.lines_number[4] == 1:
        penalty += 4
    elif player_state.lines_number[4] == 2:
        penalty += 3
    elif player_state.lines_number[4] == 3:
        penalty += 2
    elif player_state.lines_number[4] == 2:
        penalty += 1
    penalty += alone_tile_punishment(player_state)
    return penalty


def bonus_evaluation(player_state):
    # Only consider cols and sets
    cols = player_state.GetCompletedColumns()
    sets = player_state.GetCompletedSets()
    bonus = (cols * player_state.COL_BONUS) + (sets * player_state.SET_BONUS)
    return bonus


def reward_estimation(game_state, player_id):
    player_state = game_state.agents[player_id]
    first_agent = game_state.first_agent 
    bonus_initial = 0
    if player_id == first_agent:
        bonus_initial = 1
    #return  np.sum(col_row_bonus(player_state)) + _CalculateFutureBonus(game_state, player_id) - _CalculateFuturePenalty(player_state)
    return bonus_initial + bonus_evaluation(player_state) - penalty_evaluatipn(player_state)
