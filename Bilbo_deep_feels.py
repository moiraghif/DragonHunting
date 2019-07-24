from CreateBilboWorld import *
import numpy as np
from importlib import reload
import ipdb
from agents import *
import os

TOT_EPISODES=1000
MAX_EPOCH = 800

#initalize the q_table:
possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}


alpha = 0.5
gamma = 0.3
epsilon = 0.2
decay_epsilon = 0.99
rewards = []

for ep in range(TOT_EPISODES):
    #recreate the environment
    bilbo=DeepQLearningAgent(PLAYER_CHAR)
    mondo=World(WORLD_DIM,bilbo=bilbo,obstacle=True)
    #do deep Q-stuff
    game_ended=False
    epoch = 0
    current_state=bilbo.get_state()
    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
        epoch += 1
        #ipdb.set_trace()
        epsilon_fear = bilbo.fear(epsilon)
        action = bilbo.get_action(epsilon,possible_moves)
        #treasure_gone = bilbo.treasure_gone()
        reward = bilbo.reward()
        bilbo.move(inverse_possible_moves[action])()

        new_state = bilbo.get_state()
        #treasure_gone = bilbo.treasure_gone()
        game_ended = bilbo.game_ended()
        #reward = bilbo.reward()
        bilbo.add_knowledge((current_state,action,reward,new_state,game_ended))

        bilbo.train(gamma)
        #print("ep: ",ep ," epoch: ", epoch)

    ep += 1
    if ep % 1 == 0:
        #print("epoch ", epoch)
        print(mondo)
        print(bilbo.treasure_gone(),bilbo.game_ended())
        print("epoch used: ",epoch, " ep:", ep)
        print(bilbo.reward())
        os.system( 'clear' )
    epsilon *= decay_epsilon

#print(q_table)
print("Atlast the episilon value was ", epsilon)
#print(mondo)
