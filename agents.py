import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv2D,MaxPooling2D, Flatten
from collections import deque
import random
import ipdb

WORLD_DIM = 15
DRAGON_CHAR = '☠'
TREASURE_CHAR = '♚'
PLAYER_CHAR = '☺'
OBSTACLE_CHAR = "█"

MAX_MEMORY = 50000
MIN_MEMORY = 150



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
        self.action_size = len(self.actions)
        self.function_memory = [] #no more needed will remove later

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
        #self.remember_move(lambda: self.world.move_of(self, y=y, x=x))
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

    def remember_move(self,move):
        self.function_memory.append(move)

    def return_memory(self):
        return self.function_memory


class QLearningAgent(Agent):
    def learning_function(self,alpha,gamma,x_old,reward,x_new):
        return (x_old + alpha*(reward + gamma*x_new - x_old))

    def get_current_state(self):
        """
        Get the current state from the current world
        it is used as a index for the q-table
        """
        return(self.world.get_state())

    def treasure_gone(self):
        return (self.world.treasure_gone())

    def get_action(self,epsilon,q_table,possible_moves):
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

class DeepQLearningAgentImage(Agent):

    def __init__(self,char):
        super().__init__(char)
        self.q_nn = self.initialize_nn(input_shape=(WORLD_DIM,WORLD_DIM,1))
        #self.predict_nn = self.initialize_nn(input_shape=(WORLD_DIM,WORLD_DIM,1))
        #self.predict_nn.set_weights(self.q_nn.get_weights())
        self.map = {TREASURE_CHAR: '16',
                    PLAYER_CHAR: '5',
                    DRAGON_CHAR: '10',
                    OBSTACLE_CHAR: '20'}
        self.memory = deque(maxlen=MAX_MEMORY)

    def learning_function(self,alpha,gamma,x_old,reward,x_new):
        return ((1-alpha) * x_old + alpha*(reward + gamma*x_new))

    def get_state(self):
        return self.world.deep_normalized_state(self.map,image=True).reshape(WORLD_DIM,WORLD_DIM,1)

    def treasure_gone(self):
        return (self.world.treasure_gone())

    def get_qs(self,state):
        return self.q_nn.predict(state.reshape(-1,WORLD_DIM,WORLD_DIM,1))

    def get_action(self,epsilon,possible_moves):
        if np.random.uniform(0,1) < epsilon:
            action = possible_moves[self.random_action()]
        else:
            action = np.argmax(self.q_nn.predict(self.get_state().reshape(-1,WORLD_DIM,WORLD_DIM,1)))
        return(action)

    def game_ended(self):
        return(not self.world.game_state() == 0)

    def reward(self):
        return self.world.reward()
    def initialize_nn(self,input_shape):
        '''
        for the prediction do:
            env = world.create_env(d)
            state=env.reshape(-1,env.shape[0],env.shape[1],1)
            model.predict(state)
        using the -1 python should infer the batch size autmatically
        the reshape is needed to match the following shape
        (batch_size, img_height, img_width, number_of_channels)
        in our case the number_of_channels is 1 beacuse I created a B&W image
        for the fitting do:
            X_train = collection of states (the env)
            q_vals = collection of precedent (q_vals)
            X_train.reshape(-1,X_train.shape[0],X_train.shape[1],1)
            model.fit(X_train,q_vals)
        '''
        model = Sequential()
        #input shape is (DIM,DIM,1) 1 beacause they are Black&White
        model.add(Conv2D(32,(3,3), input_shape=input_shape, activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.2))
        model.add(Conv2D(16,(3,3), activation='relu'))
        model.add(MaxPooling2D((2,2)))
        model.add(Dropout(0.1)) # per evitare overfitting
        model.add(Flatten())
        model.add(Dense(16,activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer='adam')
        return model

    def add_knowledge(self, knowledge):
        self.memory.append(knowledge)

    #####################################
    ################################
    ##########BANANA STUFF
    ##############################
    #######################################



    def train(self,gamma):

        # Start training only if certain number of samples is already saved
        if len(self.memory) < MIN_MEMORY:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.memory, 32)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([memory[0] for memory in minibatch])
        current_qs_list = self.q_nn.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        next_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.q_nn.predict(next_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, next_current_state, game_ended) in enumerate(minibatch):
            # almost like with Q Learning, but we use just part of equation here
            if game_ended:
                new_q = reward
            elif (next_current_state==current_state).all():
                #any kind of obtacle which made bilbo not move
                new_q = -10 #-10 in case of obstacle
            else:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + gamma * max_future_q

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        #ipdb.set_trace()
        # Fit on all samples as one batch, log only on terminal state
        self.q_nn.fit(np.array(X), np.array(y), batch_size=32, verbose=0, shuffle=False)

        # Update target network counter every episode
        #if terminal_state:
        #    self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        #if self.target_update_counter > UPDATE_TARGET_EVERY:
        #    self.target_model.set_weights(self.model.get_weights())
        #    self.target_update_counter = 0






class DeepQLearningAgent(Agent):

    def __init__(self,char):
        super().__init__(char)
        self.q_nn = self.initialize_nn(input_shape=(3,))
        #self.predict_nn = self.initialize_nn(input_shape=(WORLD_DIM,WORLD_DIM,1))
        #self.predict_nn.set_weights(self.q_nn.get_weights())
        self.map = {TREASURE_CHAR: '16',
                    PLAYER_CHAR: '5',
                    DRAGON_CHAR: '10',
                    OBSTACLE_CHAR: '20'}
        self.memory = deque(maxlen=MAX_MEMORY)

    def learning_function(self,alpha,gamma,x_old,reward,x_new):
        return ((1-alpha) * x_old + alpha*(reward + gamma*x_new))

    def get_state(self):
        return self.world.deep_normalized_state(self.map)

    def treasure_gone(self):
        return (self.world.treasure_gone())

    def get_qs(self,state):
        return self.q_nn.predict(state)

    def get_action(self,epsilon,possible_moves):
        if np.random.uniform(0,1) < epsilon:
            action = possible_moves[self.random_action()]
        else:
            action = np.argmax(self.q_nn.predict(self.get_state().reshape(-1,3)))
        return(action)

    def game_ended(self):
        return(not self.world.game_state() == 0)

    def reward(self):
        return self.world.reward()
    def initialize_nn(self,input_shape):
        '''
        for the prediction do:
            env = world.create_env(d)
            state=env.reshape(-1,env.shape[0],env.shape[1],1)
            model.predict(state)
        using the -1 python should infer the batch size autmatically
        the reshape is needed to match the following shape
        (batch_size, img_height, img_width, number_of_channels)
        in our case the number_of_channels is 1 beacuse I created a B&W image
        for the fitting do:
            X_train = collection of states (the env)
            q_vals = collection of precedent (q_vals)
            X_train.reshape(-1,X_train.shape[0],X_train.shape[1],1)
            model.fit(X_train,q_vals)
        '''
        model = Sequential()
        #input shape is (DIM,DIM,1) 1 beacause they are Black&White
        model.add(Dense(2, input_shape=input_shape, activation='relu'))
        model.add(Dense(3,activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer='adam')
        return model

    def add_knowledge(self, knowledge):
        self.memory.append(knowledge)

    #####################################
    ################################
    ##########BANANA STUFF
    ##############################
    #######################################



    def train(self,gamma,MAX_EPOCH):

        # Start training only if certain number of samples is already saved
        if len(self.memory) < MIN_MEMORY:
            return

        # Get a minibatch of random samples from memory replay table
        minibatch = random.sample(self.memory, 32)
        # Get current states from minibatch, then query NN model for Q values
        current_states = np.array([memory[0] for memory in minibatch])
        #ipdb.set_trace()
        current_qs_list = self.q_nn.predict(current_states)

        # Get future states from minibatch, then query NN model for Q values
        # When using target network, query it, otherwise main network should be queried
        next_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.q_nn.predict(next_current_states)

        X = []
        y = []

        # Now we need to enumerate our batches
        for index, (current_state, action, reward, next_current_state, game_ended,epoch) in enumerate(minibatch):
            # almost like with Q Learning, but we use just part of equation here
            if game_ended:
                new_q = reward
            elif (next_current_state==current_state).all():
                #any kind of obtacle which made bilbo not move
                new_q = -100 #-100 in case of obstacle
            elif epoch==MAX_EPOCH:
                new_q = -2000
            else:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + gamma * max_future_q

            # Update Q value for given state
            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            # And append to our training data
            X.append(current_state)
            y.append(current_qs)

        #ipdb.set_trace()
        # Fit on all samples as one batch, log only on terminal state
        self.q_nn.fit(np.array(X), np.array(y), batch_size=32, verbose=0, shuffle=False)

        # Update target network counter every episode
        #if terminal_state:
        #    self.target_update_counter += 1

        # If counter reaches set value, update target network with weights of main network
        #if self.target_update_counter > UPDATE_TARGET_EVERY:
        #    self.target_model.set_weights(self.model.get_weights())
        #    self.target_update_counter = 0
