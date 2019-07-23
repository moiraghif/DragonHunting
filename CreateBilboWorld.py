import numpy as np
import re
#import ipdb
from agents import *

WORLD_DIM=25
DRAGON_CHAR = '☠'
DRAGON_PENALTY = 500
TREASURE_CHAR = '♚'
TREASURE_REWARD = 1000
PLAYER_CHAR = '☺'
OBSTACLE_CHAR = "█"
OBSTACLE_CONST = 0.12
OBSTACLE_PENALTY = 10

WALKING_PENALTY = 1
TOO_MUCH_WALK_PENALTY = 49

class World:
    def __init__(self, dim_x=WORLD_DIM, dim_y=None,bilbo=None,entrance=(0,0),obstacle=False):
        "Create a new World"
        self.dim_x = dim_x
        if dim_y:
            self.dim_y = dim_y
        else:
            self.dim_y = dim_x
        self.player= bilbo
        bilbo.initialize_world(self)
        self.exit = entrance
        self.world = np.array([["" for x in range(self.dim_x)]
                               for y in range(self.dim_y)])
        #making the obstacles options
        if obstacle:
        	np.random.seed(seed=1234)
        	self.insert_obstacles()
        #the position can be changed later
        self.world[round(self.dim_x / 2)-1, self.dim_x-3] = DRAGON_CHAR
        self.world[round(self.dim_x / 2)-1,self.dim_x-1] = TREASURE_CHAR
        self.world[entrance] = PLAYER_CHAR

        #this should be in agent class
        self.possible_moves = {1:'up',2:'down',3:'right',4:'left'}

    def insert_obstacles(self):
        "Insert obstacles in the new world"
        for i in range(self.dim_x):
            for j in range(self.dim_y):
            	#check if in the position chosen there is no other entity
                if ((np.random.rand() > 1 - OBSTACLE_CONST) and (self.world[i, j]=='')):
                    self.world[i, j] = OBSTACLE_CHAR

    #def get_player
    def treasure_gone(self):
    	if self.get_position(TREASURE_CHAR):
    		return 1
    	else:
    		return 0

    def game_state(self):
        "Is the game finished? In that case, return the True"
        #will be the
        #if the coin is eaten and he is at the exit again
        if ((not self.get_position(TREASURE_CHAR)) and (self.get_position(self.player.char)==self.exit)):#(get_position(self.player.char)==entrance)):
        	return 1 #means he won
        #BILBO was eaten
        elif (not self.get_position(PLAYER_CHAR)):
        	return 2 #he failed
        return 0 #continue the game please

    def reward(self):
    	game_state = self.game_state()
    	if game_state==1:
    		return TREASURE_REWARD
    	elif game_state==2:
    		return -DRAGON_PENALTY
    	else:
    		return -WALKING_PENALTY

    def is_border(self, pos):
        """Check if the cell is borderline:
        return False if not, the element otherwise"""
        # World.is_free((x, y))
        # check if cell is in the world
        if pos[0] < 0 or pos[0] >= self.dim_x or \
           pos[1] < 0 or pos[1] >= self.dim_y:
            return False
        if self.world[pos] == OBSTACLE_CHAR:
            return False
        return True

    def move(self, pos_from, pos_to):
        "Move an element if possible"
        # if the initial position is free or
        # the finishing position is obstruited
        if not self.world[pos_from] or not self.is_border(pos_to):
            return False
        # KILL bilbo
        if self.world[pos_to]==DRAGON_CHAR:
            #self.player=None
            self.world[pos_from] = ''
        	#self.world[pos_to] = DRAGON_CHAR
            return True
        #move him
        self.world[pos_from], self.world[pos_to] = "", self.world[pos_from]
        return True

    def get_position(self, char):
        "Get the position of an agent"
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                if char == self.world[y, x]:#agent.char:
                    return (y, x)
        return False

    def move_of(self, agent, x=0, y=0):
        "Move the Biblo if possible"
        pos_from = self.get_position(agent.char)
        #ipdb.set_trace()
        # check if there is the agent on the board
        if not pos_from:
            return False
        # check if the movement is in just one direction and not null
        if (x > 0 and y > 0) or x + y == 0:
            return False
        # check if there are obstacles between
        #will be checked with self.is_border() in self.move()
        # move the agent
        pos_to = (pos_from[0] + y, pos_from[1] + x)
        return self.move(pos_from, pos_to)

    def get_state(self):
        "Return the state for the q-learning"
        player_pos = self.get_position(self.player.char)
        treasure_state = self.treasure_gone()
        game_ended = False if self.game_state()==0 else True
        return(player_pos)


    def explore(self, agent):
        "Return what an agent sees"
        return self.world

    def __str__(self):
        "Convert the world into string"
        txt = "┌" + "─" * (2 * self.dim_x - 1) + "┐\n│"
        txt += "│\n│".join(map(lambda r: " ".join([str(i) if i else " "
                                                   for i in r]),
                               self.world))
        txt += "│\n" + "└" + "─" * (2 * self.dim_x - 1) + "┘"
        return re.sub(OBSTACLE_CHAR + r"\s" + OBSTACLE_CHAR,
                      3 * OBSTACLE_CHAR, txt)



if __name__=='__main__':
	mondo=World(WORLD_DIM,WORLD_DIM)
	print(mondo)

	#seems working
	for i in range(WORLD_DIM):
		#move = mondo.random_move()
		#print(move)
		mondo.action('right')
		print(mondo)

	mondo.action('down')
	print(mondo)
	mondo.action('down')
	print(mondo)
	mondo.action('down')
	print(mondo)
	mondo.action('left')
	print(mondo)
