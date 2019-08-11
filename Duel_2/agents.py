import numpy as np


class Agent:
    '''
    Creates an q-intelligent player
    '''
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
        self.actions = None
        self.qtable = None

    def alive(self):
        "Are you alive?"
        return self.healt > 0

    def fear(self):
        "Be afraid of your enemy!"
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
        "yes, it is a clojure: it returns a function"
        return [lambda: self.world.move_of(self, x=+1),
                lambda: self.world.move_of(self, x=-1),
                lambda: self.world.move_of(self, y=+1),
                lambda: self.world.move_of(self, y=-1),
                lambda: self.world.attack(self)]

    def random_action(self, ):
        "Just a random function"
        return np.random.randint(len(self.actions))

    def best_action(self, direction, distance):
        "retruns the Best action possible according to the q-table"
        return np.argmax(self.qtable[direction, distance])

    def get_action(self, last_move=False, epsilon=0):
        "The main stuff of q-learning"
        if not self.alive():
            reward, done = self.world.do_action(self, fn=None)
            return reward, done
        direction, distance = self.world.get_closer_enemy(self)
        fn = self.random_action() \
            if np.random.rand() < self.fear() * epsilon \
            else self.best_action(direction, distance)
        old_state = self.qtable[direction, distance, fn]
        reward, done = self.world.do_action(self, self.actions[fn])
        if last_move:
            reward -= 10
        new_direction, new_distance = self.world.get_closer_enemy(self)
        new = np.max(self.qtable[new_direction, new_distance])
        q_value = self.alpha * (reward + self.gamma * new - old_state)
        self.qtable[direction, distance, fn] += q_value
        return reward, done

    def save_qtable(self):
        "Save the q-tables in your HD"
        return np.save(self.qfile, self.qtable)

    def get_pos(self):
        "return a tuple (y,x)"
        return self.world.get_pos(self)

    def put_in_grave(self):
        "tranfer the dead to the graveyard"
        self.world.put_p_in_grave(self)

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
