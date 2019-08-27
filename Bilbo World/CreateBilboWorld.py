import numpy as np
import re
from PIL import Image
from agents import *

DRAGON_CHAR = '☠'
DRAGON_PENALTY = 5
TREASURE_CHAR = '♚'
TREASURE_REWARD = 10
PLAYER_CHAR = '☺'
OBSTACLE_CHAR = "█"
OBSTACLE_CONST = 0.12
OBSTACLE_PENALTY = 1

WALKING_PENALTY = 1
TOO_MUCH_WALK_PENALTY = 49

class World:
    def __init__(self, dim_x=WORLD_DIM, dim_y=None, bilbo=None, entrance=False, obstacle=False, random_spawn=False):
        "Create a new World"
        self.dim_x = dim_x
        if dim_y:
            self.dim_y = dim_y
        else:
            self.dim_y = dim_x
        self.player = bilbo
        self.rand = random_spawn
        if bilbo:
            bilbo.initialize_world(self)
        self.world = np.array([["" for x in range(self.dim_x)]
                               for y in range(self.dim_y)])
        #making the obstacles options
        if obstacle:
            np.random.seed(seed=1234)
            self.insert_obstacles()
        #the position can be changed later
        if self.rand:
            treasure_spawn = self.random_spawn()
            self.world[treasure_spawn] = TREASURE_CHAR
            dragon_spawn = self.random_spawn()
            self.world[dragon_spawn] = DRAGON_CHAR
        else:
            self.world[round(self.dim_y/2)-1, self.dim_x-3] = DRAGON_CHAR
            self.world[round(self.dim_y/2)-1, self.dim_x-1] = TREASURE_CHAR
        if bilbo:
            bilbo_entrance = self.random_spawn(entrance)
            self.world[bilbo_entrance] = PLAYER_CHAR


        #this should be in agent class
        self.possible_moves = {1:'up',2:'down',3:'right',4:'left'}

    def random_spawn(self, entrance=False):
        '''
        check if user specified an entrance/spawn point otherwise it will
        find a new point where there are no obstacles or other characters
        '''
        if not entrance:
            np.random.seed(None)
            x = np.random.randint(WORLD_DIM)
            y = np.random.randint(WORLD_DIM)
            if self.world[x,y] == '':
                return (x,y)
            return self.random_spawn()
        return entrance

    def insert_obstacles(self):
        "Insert obstacles in the new world"
        for i in range(self.dim_x):
            for j in range(self.dim_y):
            	#check if in the position chosen there is no other entity
                if ((np.random.rand() > 1 - OBSTACLE_CONST) and (self.world[i, j]=='')):
                    self.world[i, j] = OBSTACLE_CHAR

    def treasure_gone(self):
        "Checks if the treasure is on the map"
        if self.get_position(TREASURE_CHAR):
            return 1
        return 0

    def game_state(self):
        "Is the game finished? In that case, return the True"
        #will be the
        #if the coin is eaten and he is at the exit again
        if self.get_position(TREASURE_CHAR) == (-1, -1):
            #return 1 #means he won
            if self.rand:
                new_spawn = self.random_spawn()
                self.world[new_spawn] = TREASURE_CHAR
                return 1
            return 1
        #BILBO was eaten
        if self.get_position(PLAYER_CHAR) == (-1, -1):
            return 2 #he failed
        return 0 #continue the game please

    def reward(self, current_state=None, next_state=None, moving_reward=False):
        "return the reward based on the game state"
        game_state = self.game_state()
        if game_state == 1:
            return TREASURE_REWARD
        if game_state == 2:
            return -DRAGON_PENALTY

        if not moving_reward: #for q learning
            return -WALKING_PENALTY

        if (current_state==next_state).all():
            return -OBSTACLE_PENALTY
        treasure_pos = np.array(self.get_position(TREASURE_CHAR))
        #reward can either return a negative constant (-1) or 0, but to have a faster
        #convergence it's better to give him a positive reward when he gets near
        #the objective
        player_old_pos = current_state[0:2]
        player_new_pos = next_state[0:2]
        if np.sum(np.abs(player_old_pos - treasure_pos)) <= np.sum(np.abs(player_new_pos - treasure_pos)):
            return 0
        return 0 #positive if closing in to the treasure


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
        # Kill Bilbo
        if self.world[pos_to] == DRAGON_CHAR:
            self.world[pos_from] = ''
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
        return (-1,-1) #not defined (just to have a consistent state)

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
        return player_pos

    def move_dragon(self):
        action = np.random.randint(4)
        if action == 0:
            self.move_dragon_of(x=+1)
        elif action == 1:
            self.move_dragon_of(x=-1)
        elif action == 2:
            self.move_dragon_of(y=+1)
        elif action == 3:
            self.move_dragon_of(y=-1)

    def move_dragon_of(self, x=0, y=0):
        pos_from = self.get_position(DRAGON_CHAR)
        if not pos_from:
            return False
        # check if the movement is in just one direction and not null
        if (x > 0 and y > 0) or x + y == 0:
            return False
        pos_to = (pos_from[0] + y, pos_from[1] + x)

        if not self.world[pos_from] or not self.is_border(pos_to):
            return False
        # Kill Bilbo
        if self.world[pos_to] == PLAYER_CHAR or self.world[pos_to] == TREASURE_CHAR:
            return False
        self.world[pos_from], self.world[pos_to] = "", self.world[pos_from]
        return True


    def __str__(self):
        "Convert the world into string"
        txt = "┌" + "─" * (2 * self.dim_x - 1) + "┐\n│"
        txt += "│\n│".join(map(lambda r: " ".join([str(i) if i else " "
                                                   for i in r]),
                               self.world))
        txt += "│\n" + "└" + "─" * (2 * self.dim_x - 1) + "┘"
        return re.sub(OBSTACLE_CHAR + r"\s" + OBSTACLE_CHAR,
                      3 * OBSTACLE_CHAR, txt)

    def create_env(self,d):
        '''
        Creates an enviroment with the background consisting of zeros and
        everything else is mapped according to the dictionary, this is used
        for the animation with matplotlib and to create image with self.get_image
        '''
        env = np.zeros((WORLD_DIM, WORLD_DIM), dtype=np.uint8)
        if self.get_position(TREASURE_CHAR) != (-1,-1):
            env[self.get_position(TREASURE_CHAR)[0], self.get_position(TREASURE_CHAR)[1]] = d[TREASURE_CHAR]  # sets the treasure location tile
        if self.get_position(PLAYER_CHAR) != (-1,-1):
            env[self.get_position(PLAYER_CHAR)[0], self.get_position(PLAYER_CHAR)[1]] = d[PLAYER_CHAR]
        env[self.get_position(DRAGON_CHAR)[0], self.get_position(DRAGON_CHAR)[1]] = d[DRAGON_CHAR]
        obstacles = np.argwhere(self.world == OBSTACLE_CHAR)
        for coord in obstacles:
            env[coord[0]][coord[1]] = d[OBSTACLE_CHAR]
        return env

    def get_image(self, d):
        '''
        Creates an image that can be used by the deep q learning network
        if you want to resize just do .resize((dim1,dim2)) to the
        recieved object and use .show() to see it
        '''
        env = self.create_env(d)
        env = env*(255/np.max(env))
        #scale everything between 0 and 255
        img = Image.fromarray(env) #255 max color
        return img

    def deep_normalized_state(self, d, image=False):
        '''
        Creates the input ''image'' for the DQN
        if image is False the states are the coordinates of the entities
        '''
        if image:
            env = self.create_env(d)
            return env/np.max(env)
        p_pos = self.get_position(PLAYER_CHAR)
        t_pos = self.get_position(TREASURE_CHAR)
        d_pos = self.get_position(DRAGON_CHAR)
        is_down_free = 1 if self.is_border((p_pos[0] + 1, p_pos[1])) else 0
        is_up_free = 1 if self.is_border((p_pos[0] - 1, p_pos[1])) else 0
        is_right_free = 1 if self.is_border((p_pos[0], p_pos[1] + 1)) else 0
        is_left_free = 1 if self.is_border((p_pos[0] + 1, p_pos[1] - 1)) else 0
        state = [p_pos[0] - t_pos[0], p_pos[1] - t_pos[1],
                 p_pos[0] - d_pos[0], p_pos[1] - d_pos[1],
                 is_down_free, is_up_free, is_right_free, is_left_free]
        return np.array(state)

    def set_bilbo(self, bilbo):
        self.player = bilbo
        bilbo.initialize_world(self)
        #if self.rand:
        #    treasure_spawn = self.random_spawn()
        #    self.world[treasure_spawn] = TREASURE_CHAR
        #    dragon_spawn = self.random_spawn()
        #    self.world[dragon_spawn] = DRAGON_CHAR
        #else:
        #    self.world[round(self.dim_y/2)-1, self.dim_x-3] = DRAGON_CHAR
        #    self.world[round(self.dim_y/2)-1, self.dim_x-1] = TREASURE_CHAR
        for x in range(self.dim_x):
            for y in range(self.dim_y):
                if self.world[y,x] == PLAYER_CHAR or self.world[y,x] == DRAGON_CHAR:
                    self.world[y,x] = ''
        dragon_spawn = self.random_spawn()
        self.world[dragon_spawn] = DRAGON_CHAR

        bilbo_entrance = self.random_spawn()
        self.world[bilbo_entrance] = PLAYER_CHAR


if __name__ == '__main__':
    print('This file has the World class in it!')
