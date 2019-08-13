import numpy as np
import constants


class Agent:
    '''
    Creates an q-intelligent player
    '''
    def __init__(self, char):
        "Create a new agent"
        self.char = char
        self.type = 'd' if self.char==constants.DRAGON_CHAR else 'p' #type can be 'p'=player or 'd'=dragon
        self.world = None
        self.healt = 10 if self.type == 'p' else 15
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
        self.qtable = np.array([[[[[np.random.rand()
                                    for _ in range(len(self.actions))]
                                   for player_1_dist in range(2)]
                                  for player_1_direction in range(4)]
                                 for player_2_dist in range(2)]
                                for player_2_direction in range(4)]) \
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

    def best_action(self, current_state):
        "retruns the Best action possible according to the q-table"
        return np.argmax(self.qtable[current_state])

    def get_current_state(self):
        return self.world.get_current_state(self)

    def get_action(self, last_move=False, epsilon=0):
        "The main stuff of q-learning"
        if not self.alive():
            current_state = self.get_current_state()
            reward, done = self.world.do_action(self, fn=None)
            self.qtable[current_state] -= self.gamma*reward
            if self.type == 'p':
                for p in self.world.players.keys():
                    if p.type == 'p' and p.char!=self.char:
                        p.qtable[p.get_current_state()] -= p.gamma*reward
            return reward, done
        current_state = self.get_current_state()
        fn = self.random_action() \
            if np.random.rand() < self.fear() * epsilon \
            else self.best_action(current_state)
        old_q = self.qtable[current_state][fn]
        reward, done = self.world.do_action(self, self.actions[fn])
        if last_move:
            reward -= 10
        new_state = self.get_current_state()
        new = np.max(self.qtable[new_state])
        q_value = self.alpha * (reward + self.gamma * new - old_q)
        if epsilon!=0: #don't save in the test phase
            self.qtable[current_state][fn] += q_value
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
