import numpy as np
from CreateBilboWorld import *

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

    def __str__(self):
        return self.char

    def __eq__(self, other):
        "Check if two agents are the same"
        try:
            return self.char == other.char
        except NameError:
            return False


class QLearningAgent(Agent):
    def learning_function(self):
        pass


class DeepQLearningAgent(Agent):
    def learning_function(self):
        pass
