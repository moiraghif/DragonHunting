import os
from collections import deque
import random
import numpy as np
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense
import constants


class Agent:
    '''
    Creates an q-intelligent player
    '''
    def __init__(self, char):
        "Create a new agent"
        self.char = char
        self.type = 'd' if self.char == constants.DRAGON_CHAR else 'p' #type can be 'p'=player or 'd'=dragon
        self.world = None
        self.healt = 10 if self.type == 'p' else 12
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


"""
**********************************
DEEP LEARNING AGENT FROM NOW ON!!
**********************************
"""

class DeepAgent(Agent):
    def __init__(self, char):
        super().__init__(char)
        self.qtable = None
        self.state_dim = 5
        self.action_size = 5
        self.qNN = self.init_model()
        self.memory = deque(maxlen=constants.MAX_MEMORY)
        self.hight_reward_memory = deque(maxlen=constants.MAX_MEMORY)

    def init_model(self):
        if os.path.isfile('./models/deep_model_'+self.char+'.model'):
            print('*******************************************')
            print('*******************************************')
            print('*Found an existent model, loading that one*')
            print('*******************************************')
            print('*******************************************')
            model = load_model('./models/deep_model_'+self.char+'.model')

            print(model.summary())
            return model

        print('***************************************')
        print('***************************************')
        print('*No existent model, creating a new one*')
        print('***************************************')
        print('***************************************')
        input_shape = (self.state_dim, )
        model = Sequential()

        model.add(Dense(2*self.state_dim, input_shape=input_shape, activation='relu'))
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

    def get_current_deep_state(self):
        return self.world.deep_learning_state(self)

    def get_qs(self, state):
        "return the predicted q-values from the NN"
        return self.qNN.predict(state.reshape(-1, self.state_dim))[0]

    def best_action(self, current_state):
        "retruns the Best action possible according to the q-NN"
        return np.argmax(self.get_qs(current_state))


    def get_action(self, last_move=False, epsilon=0):
        "return the action based on the epsilon-greedy policy"
        if not self.alive():
            reward, done = self.world.do_action(self, fn=None)
            for p in self.world.players.keys():
                if self.type == 'd':
                    continue
                if p.char != self.char and p.type == self.type:
                    p_state = p.get_current_deep_state()
                    action = 4
                    p.add_high_knowledge((p_state, action, reward, [0,0,0,0,0], True))
            return self.get_current_deep_state(), None, reward, np.array([0,0,0,0,0]), done
        current_state = self.get_current_deep_state()
        fn = self.random_action() \
            if np.random.rand() < epsilon \
            else self.best_action(current_state)
        reward, done = self.world.do_action(self, fn=self.actions[fn])
        if reward == -1000 or reward == 2000:
            for p in self.world.players.keys():
                if self.type == 'd':
                    continue
                if p.char != self.char and p.type == self.type:
                    p_state = p.get_current_deep_state()
                    action = 4
                    p.add_high_knowledge((p_state, action, reward, [0,0,0,0,0], True))
        next_state = self.get_current_deep_state()

        return current_state, fn, reward, next_state, done

    def train(self):
        if len(self.memory) < constants.MIN_MEMORY*2:
            return

        if len(self.hight_reward_memory) < constants.MIN_MEMORY:
            normal_size = 32 - len(self.hight_reward_memory)
            high_mem_size = len(self.hight_reward_memory)
        else:
            normal_size = 16
            high_mem_size = 16
        # random elements from memory
        minibatch = random.sample(self.memory, normal_size)
        if high_mem_size > 0:
            minibatch.extend(random.sample(self.hight_reward_memory, high_mem_size))

        current_states = np.array([memory[0] for memory in minibatch])
        current_qs_list = self.qNN.predict(current_states)

        next_states = np.array([memory[3] for memory in minibatch])
        future_qs_list = self.qNN.predict(next_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, next_current_state, game_ended) in enumerate(minibatch):
            # almost like with Q Learning, but we use just part of equation here
            if game_ended:
                new_q = reward
            elif (next_current_state == current_state).all():
                #any kind of obtacle which made bilbo not move
                new_q = -100
            else:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + self.gamma * max_future_q

            # Aggiorna i q-value
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            y.append(current_qs)

        self.qNN.fit(np.array(X), np.array(y), batch_size=32, verbose=0, shuffle=False)

    def save_NN(self):
        model_name = './models/deep_model_'+self.char+'.model'
        save_model(self.qNN, model_name)

    def reset(self):
        self.healt = 10 if self.type == 'p' else 12
        self.temp_healt = self.healt
