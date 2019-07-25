import constants
import numpy as np


class Agent:
    def __init__(self, char, qfile):
        "Create a new agent"
        self.char = char
        self.world = None
        self.healt = 10
        self.temp_healt = self.healt
        self.qfile = qfile
        # a list of actions (functions) that the Agent can do
        self.actions = [action for action in self.generate_actions()]
        self.alpha = 0.75
        self.gamma = 0.9

    def initialize_world(self, world, create_qtable):
        "set the world and the initial qtable (with random values)"
        self.world = world
        self.qtable = np.array(
            [[[[[np.random.rand()
                 for _ in range(len(self.actions))]
                for xe in range(self.world.dim_x)]
               for ye in range(self.world.dim_y)]
              for xa in range(self.world.dim_x)]
             for ya in range(self.world.dim_y)]) \
            if create_qtable else np.load(self.qfile + ".npy")
        # reverse of:
        # action, (enemy_position), (self_position)

    def generate_actions(self):
        "Generate the function to mode the agent"
        # yes, it is a clojure: it returns a function
        yield lambda: self.world.move_of(self, -1, 0)
        yield lambda: self.world.move_of(self, +1, 0)
        yield lambda: self.world.move_of(self, 0, -1)
        yield lambda: self.world.move_of(self, 0, +1)
        yield lambda: self.world.attack(self)

    def random_action(self):
        return np.random.randint(len(self.actions))

    def best_action(self):
        other = self.world.get_enemy(self)
        return np.argmax(self.qtable[(*self.world.players[self]),
                                     (*self.world.players[other])])

    def get_action(self, last_move=False):
        fn = self.random_action() \
            if np.random.rand() < constants.EPSILON \
            else self.best_action()
        self_pos = tuple([*self.world.players[self]])
        other_pos = tuple([*self.world.players[self.world.get_enemy(self)]])
        old = self.qtable[(*self_pos), (*other_pos), fn]
        reward, done = self.world.do_action(self, self.actions[fn])
        if last_move and self.world.get_enemy(self).healt > 0:
            reward -= 10
        if self.healt != self.temp_healt:
            reward -= 1
            self.temp_healt = self.healt
        new = max(self.qtable[(*self_pos), (*other_pos)])
        q_value = self.alpha * (reward + self.gamma * new - old)
        self.qtable[(*self_pos), (*other_pos), fn] += q_value
        return reward, done

    def save_qtable(self):
        return np.save(self.qfile, self.qtable)

    def __str__(self):
        return self.char

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        "Check if two agents are the same"
        try:
            return self.char == other.char
        finally:
            return False
