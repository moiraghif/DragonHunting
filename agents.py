import numpy as np
#import ipdb

DRAGON_CHAR = '☠'
TREASURE_CHAR = '♚'
PLAYER_CHAR = '☺'
OBSTACLE_CHAR = "█"

class Agent:
    def __init__(self, char):
        "Create a new agent"
        self.char = char
        self.world = None
        # a list of possible movements that the agent can do
        movements = list(map(self.move,
                             ["up", "down", "left", "right"]))
        # a list of actions (functions) that the Agent can do
        self.actions = movements + []

    def initialize_world(self, world):
        self.world = world

    def get_pos(self):
        return(self.world.get_position(self.char))

    def move(self, d):
        "Generate the function to mode the agent"
        # yes, it is a clojure: it returns a function
        x, y = 0, 0
        if d == "right":
            x = +1
        elif d == "left":
            x = -1
        elif d == "down":
            y = +1
        elif d == "up":
            y = -1
        else:
            raise Exception(str(d) + "is not a valid direction")
        return lambda: self.world.move_of(self, y=y, x=x)

    def random_action(self):
        #mi serve il nome dell'azione
        movements = ["up", "down", "left", "right"]
        return(movements[np.random.randint(len(self.actions))])

    def fear(self,epsilon):
        '''
        return a new epsilon based also on a fear factor, if it's far from the
        dragon then the the value is near epsilon (so low fear), if it's near
        dragon then the value increases at most to 2*epsilon (when the dragon
        is nearby)
        '''
        dragon_pos = self.world.get_position(DRAGON_CHAR)
        self_pos = self.get_pos()
        dist = np.sqrt((dragon_pos[0]-self_pos[0])^2 + (dragon_pos[1]-self_pos[1])^2)
        return epsilon/(dist/(1+dist))

    def __str__(self):
        return self.char

    def __eq__(self, other):
        "Check if two agents are the same"
        try:
            return self.char == other.char
        except NameError:
            return False


class QLearningAgent(Agent):
    def learning_function(self,alpha,gamma,x_old,reward,x_new):
        return ((1-alpha) * x_old + alpha*(reward + gamma*x_new))

    def get_current_state(self):
        """
        This will be changed in a near future in order to get
        the current state directly from the environment which
        can be different from the position
        """
        return(self.world.get_position(self.char))

    def treasure_gone(self):
        return (self.world.treasure_gone())

    def get_action(self,epsilon,q_table,possible_moves):
        #ipdb.set_trace()
        if np.random.uniform(0,1) < epsilon:
            action = possible_moves[self.random_action()]
        else:
            action = np.argmax(q_table[self.get_current_state(),
                                       self.treasure_gone()])
        return(action)
    def game_ended(self):
        return(not self.world.game_state() == 0)

    def reward(self):
        return self.world.reward()

class DeepQLearningAgent(Agent):
    def learning_function(self):
        pass
