import constants
import numpy as np


class Agent:
    def __init__(self, char):
        "Create a new agent"
        self.char = char
        self.world = None
        self.healt = 10
        self.temp_healt = self.healt
        self.qfile = "qtable_" + self.char
        # a list of actions (functions) that the Agent can do
        self.alpha = 0.5
        self.gamma = 0.9

    def alive(self):
        return self.healt > 0

    def fear(self):
        try:
            return (1 / self.healt) * \
                ((1 + self.world.get_dist_to_enemies(self)) / self.world.get_dist_to_enemies(self))
        except ZeroDivisionError:
            return 0

    def initialize_world(self, world, create_qtable):
        "set the world and the initial qtable (with random values)"
        self.world = world
        self.actions = list(self.generate_actions())
        self.qtable = np.array([[[np.random.rand()
                                  for _ in range(len(self.actions))]
                                 for distance in range(2)]
                                for direction in range(4)]) \
            if create_qtable else np.load(self.qfile + ".npy")
        # enemy direction, enemy distance, action

    def generate_actions(self):
        # yes, it is a clojure: it returns a function
        return [lambda: self.world.move_of(self, x=+1),
                lambda: self.world.move_of(self, x=-1),
                lambda: self.world.move_of(self, y=+1),
                lambda: self.world.move_of(self, y=-1),
                lambda: self.world.attack(self)]

    def random_action(self, ):
        return np.random.randint(len(self.actions))

    def best_action(self, direction, distance):
        return np.argmax(self.qtable[direction, distance])

    def get_action(self, last_move=False,epsilon=0):
        if not self.alive():
            return 0, True
        direction, distance = self.world.get_closer_enemy(self)
        fn = self.random_action() \
            if np.random.rand() < self.fear() * epsilon \
            else self.best_action(direction, distance)
        old_state = self.qtable[direction, distance, fn]
        reward, done = self.world.do_action(self, self.actions[fn])
        if last_move:
            reward -= 10
        new = np.max(self.qtable[direction, distance])
        q_value = self.alpha * (reward + self.gamma * new - old_state)
        self.qtable[direction, distance, fn] += q_value
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
