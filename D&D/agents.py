import constants
import keras
import numpy as np


class Agent:
    def __init__(self, char, party):
        "Create a new agent"
        self.char = char
        self.party = party
        self.world = None
        self.hp = np.random.randint(90, 111)
        self.max_hp = self.hp
        self.attack = np.random.randint(12) + 1
        self.defense = np.random.randint(20) + 1
        # a list of actions (functions) that the Agent can do
        directions = ["up", "down", "left", "right"]
        self.actions = {name: action
                        for action_in_dir in map(self.generate_actions,
                                                 directions)
                        for name, action in action_in_dir}
        self.actions_labels = np.array(list(self.actions.keys()))
        self.actions_labels = [str(i) for i in range(len(self.actions))]
        self.learning_rate = 0.9
        self.brain = keras.models.Sequential()

    def build_brain(self):
        input_shape = (self.world.world.shape[0], )
        self.brain.add(keras.layers.Dense(20*20,
                                          activation="relu",
                                          input_shape=input_shape))
        self.brain.add(keras.layers.Dense(15*15,
                                          activation="relu"))
        self.brain.add(keras.layers.Dense(len(self.actions),
                                          activation="linear"))
        self.brain.compile(optimizer="adam", loss="mse")
        keras.utils.to_categorical(self.actions_labels)
        self.brain.fit(self.world_to_int(), self.actions_labels)

    def initialize_world(self, world):
        "set the variable world"
        self.world = world
        self.build_brain()

    def world_to_int(self):
        return np.vectorize(lambda x: ord(x) if x
                            else 0)(self.world.world.copy())

    def generate_actions(self, d):
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
        return [("move " + d,
                 lambda: self.world.move_of(self, y=y, x=x)),
                ("attack " + d,
                 lambda: self.world.attack(self, y=y, x=x))]

    def random_action(self):
        return self.actions[np.random.choice(list(self.actions.keys()))]

    def best_action(self):
        return self.brain.predict(self.world_to_int())

    def get_action(self):
        fn = self.random_action() if np.random.rand() < constants.EPSILON \
            else self.best_action()
        return self.world.do_action(self, fn)

    def __str__(self):
        return self.char

    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        "Check if two agents are the same"
        try:
            return self.char == other.char
        except NameError:
            return False
