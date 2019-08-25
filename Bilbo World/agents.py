import os
from collections import deque
import random
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense
import numpy as np


WORLD_DIM = 15
DRAGON_CHAR = '☠'
TREASURE_CHAR = '♚'
PLAYER_CHAR = '☺'
OBSTACLE_CHAR = "█"

MAX_MEMORY = 5000
MIN_MEMORY = 16



class Agent:
    "Just an Agent(or as we call it Bilbo) which moves on the GridWorld"
    def __init__(self, char):
        "Create a new agent"
        self.char = char
        self.world = None
        # a list of possible movements that the agent can do
        movements = list(map(self.move,
                             ["up", "down", "left", "right"]))
        # a list of actions (functions) that the Agent can do
        self.actions = movements + []
        self.action_size = len(self.actions)

    def initialize_world(self, world):
        "Initialize the world for Bilbo"
        self.world = world

    def get_pos(self):
        "return the position (y,x) of the agent on the gridWorld"
        return self.world.get_position(self.char)

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
        #self.remember_move(lambda: self.world.move_of(self, y=y, x=x))
        return lambda: self.world.move_of(self, y=y, x=x)

    def random_action(self):
        "return any of the possible actions"
        #mi serve il nome dell'azione
        movements = ["up", "down", "left", "right"]
        return movements[np.random.randint(len(self.actions))]

    def fear(self,epsilon):
        '''
        return a new epsilon based also on a fear factor, if it's far from the
        dragon then the the value is near epsilon (so low fear), if it's near
        dragon then the value increases at most to 2*epsilon (when the dragon
        is nearby)
        '''
        dragon_pos = self.world.get_position(DRAGON_CHAR)
        self_pos = self.get_pos()
        dist = np.sqrt((dragon_pos[0]-self_pos[0])**2 + (dragon_pos[1]-self_pos[1])**2)
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
    "Agent generalization for q learning"
    def learning_function(self, alpha, gamma, x_old, reward, x_new):
        "the q-function"
        return x_old + alpha*(reward + gamma*x_new - x_old)

    def get_current_state(self):
        """
        Get the current state from the current world
        it is used as a index for the q-table
        """
        return self.world.get_state()

    def treasure_gone(self):
        "check is the treasure is in the map"
        return self.world.treasure_gone()

    def get_action(self, epsilon, q_table, possible_moves):
        "gets the action using epsilon-greedy policy"
        if np.random.uniform(0, 1) < epsilon:
            action = possible_moves[self.random_action()]
        else:
            action = np.argmax(q_table[self.get_current_state()])
        return action

    def game_ended(self):
        "checks if the game has ended"
        return not self.world.game_state() == 0

    def reward(self):
        "return the reward for the action taken by the agent"
        return self.world.reward()

class DeepQLearningAgentImage(Agent):
    '''
    The DeepQLearningAgent whose input is the matrix of the world
    and this matrix represent the image of the world in Black and White
    the image has the pixel as the dimension of the world
    '''
    def __init__(self,char):
        "initialized the main class"
        super().__init__(char)
        self.state_shape = 8
        self.q_nn = self.initialize_nn(input_shape=(self.state_shape,))
        self.map = {TREASURE_CHAR: '16',
                    PLAYER_CHAR: '5',
                    DRAGON_CHAR: '10',
                    OBSTACLE_CHAR: '20'}
        self.memory = deque(maxlen=MAX_MEMORY)
        self.hight_reward_memory = deque(maxlen=MAX_MEMORY)

    def learning_function(self, alpha, gamma, x_old, reward, x_new):
        "the Q-learning function"
        return (1-alpha) * x_old + alpha*(reward + gamma*x_new)

    def get_state(self):
        "reshapes the state for the NN"
        #return self.world.deep_normalized_state(self.map, image=True).reshape(WORLD_DIM, WORLD_DIM, 1)
        return self.world.deep_normalized_state(self.map, image=False)

    def treasure_gone(self):
        "checks if the treasure still exists"
        return self.world.treasure_gone()

    def get_qs(self, state):
        "return the predicted q-values from the NN"
        return self.q_nn.predict(state.reshape(-1, self.state_shape))[0]

    def get_action(self, epsilon, possible_moves):
        "return the action based on the epsilon-greedy policy"
        if np.random.uniform(0, 1) < epsilon:
            return possible_moves[self.random_action()]
        action = np.argmax(self.get_qs(self.get_state()))
        return action

    def game_ended(self):
        "simply checks if the game has ended or not"
        return not self.world.game_state() in [0,1]

    def reward(self, current_state, next_state):
        "returns the reward that bilbo gets for the action"
        return self.world.reward(current_state, next_state, moving_reward=True)

    def initialize_nn(self, input_shape):
        '''
        for the prediction do:
            env = world.create_env(d)
            state=env.reshape(-1,self.state_shape)
            model.predict(state)
        using the -1 python should infer the batch size autmatically
        '''
        if os.path.isfile('./models/deep_model_'+str(WORLD_DIM)+'.model'):
            print('*******************************************')
            print('*******************************************')
            print('*Found an existent model, loading that one*')
            print('*******************************************')
            print('*******************************************')
            model = load_model('./models/deep_model_'+str(WORLD_DIM)+'.model')

            print(model.summary())
            return model

        print('***************************************')
        print('***************************************')
        print('*No existent model, creating a new one*')
        print('***************************************')
        print('***************************************')
        model = Sequential()

        model.add(Dense(2*self.state_shape, input_shape=input_shape, activation='relu'))
        model.add(Dense(8, activation='relu'))
        #output layer
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer='adam')
        print(model.summary())
        return model

    def add_knowledge(self, knowledge):
        "appends the game state in the memory thus adding knowledge for DQN"
        self.memory.append(knowledge)

    def add_high_knowledge(self, knowledge):
        "appends the game state in the memory thus adding knowledge for DQN"
        self.hight_reward_memory.append(knowledge)

    def train(self, gamma):
        "Trains the NN with minibatch technique with a little twist"
        # Inizio training solo dopo almeno MIN_MEMORY elementi
        if len(self.memory) < MIN_MEMORY or len(self.hight_reward_memory) < MIN_MEMORY:
            return

        # random elements from memory
        minibatch = random.sample(self.memory, 16)
        minibatch.extend(random.sample(self.hight_reward_memory,16))

        current_states = np.array([memory[0] for memory in minibatch])
        current_qs_list = self.q_nn.predict(current_states)

        next_states = np.array([memory[3] for memory in minibatch])
        future_qs_list = self.q_nn.predict(next_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, next_current_state, game_ended) in enumerate(minibatch):
            # almost like with Q Learning, but we use just part of equation here
            if game_ended:
                new_q = reward
            elif (next_current_state == current_state).all():
                #any kind of obtacle which made bilbo not move
                new_q = -7
            else:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + gamma * max_future_q

            # Aggiorna i q-value
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.q_nn.fit(np.array(X), np.array(y), batch_size=32, verbose=0, shuffle=False)
