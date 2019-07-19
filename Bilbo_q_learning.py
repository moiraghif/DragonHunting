from CreateBilboWorld import *
import numpy as np
from importlib import reload
import ipdb
from agents import *
from IPython.display import clear_output
from time import sleep
import os

def clear():
  os.system( 'clear' )

TOT_EPISODES=100
MAX_EPOCH = 500

#initalize the q_table:
possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}
q_table={}
for y in range(WORLD_DIM):
  for x in range(WORLD_DIM):
    for exist_reward in [0,1]:
        q_table[(y,x),exist_reward]=[0,0,0,0]


alpha = 0.5
gamma = 0.3
epsilon = 0.2
decay_epsilon = 0.99
rewards = []

for ep in range(TOT_EPISODES):
  #recreate the environment
  bilbo=QLearningAgent(PLAYER_CHAR)
  mondo=World(WORLD_DIM,bilbo=bilbo,obstacle=True)
  #print(World)
  #do Q-stuff
  #print(bilbo.get_pos())
  game_ended=False
  epoch = 0
  try:
    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
      epoch += 1
      epsilon_fear = bilbo.fear(epsilon)
      action = bilbo.get_action(epsilon_fear,q_table,possible_moves)
      current_state = bilbo.get_current_state()
      treasure_gone = bilbo.treasure_gone()

      old_q_val = q_table[current_state,treasure_gone][action]
      bilbo.move(inverse_possible_moves[action])()

      new_state = bilbo.get_current_state()
      treasure_gone = bilbo.treasure_gone()
      game_ended = bilbo.game_ended()
      reward = bilbo.reward()

      if reward == -50:
        new_q_val = reward
      elif epoch == MAX_EPOCH:
        reward = -50
      elif new_state==current_state:
        #any kind of obtacle which made bilbo not move
        new_q_val = -10 #-10 in case of obstacle
      else:
        next_q_val = np.max(q_table[new_state,treasure_gone])
        new_q_val = bilbo.learning_function(alpha,gamma,old_q_val,reward,next_q_val)

      q_table[current_state,treasure_gone][action] = new_q_val


    if ep % 10 == 0:
      #print("epoch ", epoch)
      #print(mondo)
      print(bilbo.get_current_state(),bilbo.treasure_gone(),bilbo.game_ended())
      print("epoch used: ",epoch, " ep:", ep)
      print(bilbo.reward())
      clear()
      #sleep(1)
    #sleep(1)
    epsilon *= decay_epsilon
  except:
    #had some issues with some index can be removed later
    #kept it in case there were some errors it's easier to debug
    print("Ops! Something went wrong!")
    ipdb.set_trace()


#print(q_table)
print("Atlast the episilon value was ", epsilon)
#print(mondo)
