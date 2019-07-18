import numpy as np
import re
#import ipdb

WORLD_DIM=7 
DRAGON_CHAR = '☠'
TREASURE_CHAR = '♚'
PLAYER_CHAR = '☺'
OBSTACLE_CHAR = "█"
OBSTACLE_CONST = 0.12

class World:
    def __init__(self, dim_x=WORLD_DIM, dim_y=WORLD_DIM,bilbo=None,entrance=(0,0),obstacle=False):
        "Create a new World"
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.player= bilbo
        self.exit  = entrance
        self.world = np.array([["" for x in range(self.dim_x)]
                               for y in range(self.dim_y)])
        #making the obstacles options
        if obstacle:
        	self.insert_obstacles()
        #the position can be changed later
        self.world[round(self.dim_x / 2)-1, round(self.dim_x / 2)+1] = DRAGON_CHAR
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

    def game_ended(self):
        "Is the game finished? In that case, return the True"
        #will be the 
        #if the coin is eaten and he is at the exit again
        if ((not self.get_position(TREASURE_CHAR)) and (self.get_position(PLAYER_CHAR)==self.exit)):#(get_position(self.player.char)==entrance)):
        	return 1 #means he won
        #BILBO was eaten
        elif (not self.get_position(PLAYER_CHAR)):
        	return 2 #he failed
        elif (not self.get_position(TREASURE_CHAR)):
        	return 3 #he got the coin (we'll see if needed)
        return 0 #continue the game please 

    def is_border(self, pos):
        """Check if the cell is borderline:
        return False if not, the element otherwise"""
        # World.is_free((x, y))
        # check if cell is in the world
        if pos[0] < 0 or pos[0] >= self.dim_x or \
           pos[1] < 0 or pos[1] >= self.dim_y:
            return False
        return True
    #    return self.world[pos] == ""

    def move(self, pos_from, pos_to):
        "Move an element if possible"
        # if the initial position is free or
        # the finishing position is obstruited
        if not self.world[pos_from] or not self.is_border(pos_to):
            return False
        # KILL bilbo 
        if self.world[pos_to]==DRAGON_CHAR:
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

    def move_if(self, x=0, y=0):
        "Move the Biblo if possible"
        pos_from = self.get_position(PLAYER_CHAR)
        #ipdb.set_trace()
        # check if there is the agent on the board
        if not pos_from:
            return False
        # check if the movement is in just one direction and not null
        if (x > 0 and y > 0) or x + y == 0:
            return False
        # check if there are obstacles between
        #if (self.world[pos_from[0]:(pos_from[0] + y + 1),
        #               pos_from[1]:(pos_from[1] + x + 1)
        #               ] != "" ).any():
        #the vector above contains also the agent itselt in pos (pos_from[0],pos_from[1])
        #thus we need to give clearance even in case it has both '' and agent.__str__()
        #if (not np.in1d(self.world[pos_from[0]:(pos_from[0] + y + 1),
        #                           pos_from[1]:(pos_from[1] + x + 1)],
        #                ["",agent.__str__()]).all()):
        #    return False
        # move the agent
        pos_to = (pos_from[0] + y, pos_from[1] + x)
        return self.move(pos_from, pos_to)

    def action(self,action):
    	if action=='right': #right
    		self.move_if(1,0)
    	if action=='left':
    		self.move_if(-1,0)
    	if action=='up':
    		self.move_if(0,-1)
    	if action=='down':
    		self.move_if(0,1)


    def explore(self, agent):
        "Return what an agent sees"
        return self.world

    def random_move(self):
    	return self.possible_moves[np.random.randint(1,5)]

    def __str__(self):
        "Convert the world into string"
        txt = "┌" + "─" * (2 * self.dim_x - 1) + "┐\n│"
        txt += "│\n│".join(map(lambda r: " ".join([str(i) if i else " "
                                                   for i in r]),
                               self.world))
        txt += "│\n" + "└" + "─" * (2 * self.dim_x - 1) + "┘"
        return re.sub(OBSTACLE_CHAR + r"\s" + OBSTACLE_CHAR,
                      3 * OBSTACLE_CHAR, txt)



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
print(mondo.game_ended())