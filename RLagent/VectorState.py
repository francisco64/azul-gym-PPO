from template import Agent
import random
from Azul.azul_model import AzulGameRule as GameRule
import numpy as np
from copy import deepcopy

def vectorizeAction(action):
    _,fid,tg=action
    #vector of 5 elements, 1 refering to the factory id taken, all 0 if taken from pool
    factoryVector=5*[0]
    factoryOrPool=[int(fid==-1)]#0 if factory 1 if pool
    
    if fid!=-1:
        factoryVector[action[1]]=1
    
    #vector of 5 elements, 1 refering to the tile type taken
    tileTypeVector=5*[0]
    tileTypeVector[tg.tile_type]=1
    
    #vector of 5 elements, 1 refering to the pattern the tile goes to
    patternLineVector=5*[0]
    patternLineVector[tg.pattern_line_dest]=1
    
    multi_hot_action=factoryOrPool+factoryVector+tileTypeVector+patternLineVector

    return np.array(multi_hot_action)

def descreteAction(action):
    multihot_vector=vectorizeAction(action)
    

    radixes = [6, 5, 5]
    sections = np.split(multihot_vector, np.cumsum(radixes)[:-1])
    digits = [np.where(section==1)[0][0] for section in sections]
    number = 0
    for radix, digit in zip(radixes, digits):
        number = number * radix + digit

    
    return number




def descreteToMultihot(number):
    radixes = [6, 5, 5]
    digits = []
    for radix in reversed(radixes):
        number, digit = divmod(number, radix)
        digits.insert(0, digit)
    multihot = []
    for radix, digit in zip(radixes, digits):
        section = np.zeros(radix)
        section[digit] = 1
        multihot.extend(section)
        
    return np.array(multihot)

#mode="similarity" or mode="random" to chose action if not in valid actions
def selectActionFromInteger(discreteAction,actions,mode="similarity"):
    multi_hot_chosen=descreteToMultihot(discreteAction)

    similarities=[]
    for action in actions:
        multi_hot_action=vectorizeAction(action)
        if (multi_hot_chosen == multi_hot_action).all():
            return action
        else:
            if mode=="similarity":
                similarities.append(np.dot(multi_hot_chosen,multi_hot_action  ))
            else:
                continue
    if mode=="similarity":
        return actions[np.argmax(similarities)]
    elif mode=="random":
        return random.choice(actions)
     
def toVectorState(game_state,playerId):
    playerStateVector=stateAgentVector(game_state,playerId)
    opponentStateVector=stateAgentVector(game_state,1-playerId)

    tileType,count=np.unique(game_state.bag_used,return_counts=True)
        
    bagUsedVector=5*[0]
    for i,type in enumerate(tileType):
        bagUsedVector[type]=count[i]/20

    #list of dictionaries with the number of tiles in each factory (fid in actions is the position of the factory in this list)
    tilesInFactory=[f.tiles for f in game_state.factories] #{<Tile.BLUE: 0>: 2, <Tile.YELLOW: 1>: 0, <Tile.RED: 2>: 0, <Tile.BLACK: 3>: 1, <Tile.WHITE: 4>: 1}
    factoryVector=[]
    for factory in tilesInFactory:
        factoryVector+=[i/4 for i in factory.values()]
    
    tilesInCenterPool=game_state.centre_pool.tiles#dictionary with the number of tiles in the center pool
    
    poolVector=[i/20 for i in tilesInCenterPool.values()]
        
    stateVector=playerStateVector+opponentStateVector+bagUsedVector+factoryVector+poolVector

    return np.array(stateVector)
    

def stateAgentVector(game_state,agentId):
            agent=game_state.agents[agentId]
            
            
            patternLineNumber=agent.lines_number
            
            
            patternLineTile=agent.lines_tile#-1 if no tile, Tile Object {IntEnum} (Blue,0) (Yellow,1) (red,2) (black,3) (white,4)
            
            patternLineValueHot=[]
            for i,numberTiles in enumerate(patternLineNumber):
                valueHotTile=5*[0]
                valueHotTile[patternLineTile[i]]=numberTiles/(i+1)#normalized 
                patternLineValueHot+=valueHotTile
                
            
            #tile.value retrieves the number
            
            grid=agent.grid_state #5x5 binary numpy array tile not tile
            
            grid_vector=list(grid.flatten())
            
            
            number_of_specific_tile_type_in_grid=agent.number_of #dictionary that states how many tiles there are in the grid for each color
            
            num_tiles_grid_vector=[i/5 for i in number_of_specific_tile_type_in_grid.values()]#normalized
            
            
            floorLine=agent.floor #binary list of 7, tile not tile #NOTHING TO DO, USE AS FEATURE VECTOR
            
        
            #agent state
            auxiliarAgentState=deepcopy(agent)
            
            #won't be used as  vector but will be useful as a reward
            #ScorePartialRound=auxiliarAgentState.ScoreRound()[0]#can be good for the reward function. Each round registers the score of the grid
    
            
            
            
            stateAgentVector=patternLineValueHot+grid_vector+num_tiles_grid_vector+floorLine
        
            return stateAgentVector
