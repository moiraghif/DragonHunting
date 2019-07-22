from CreateBilboWorld import *
import numpy as np
#from importlib import reload
import ipdb
from agents import *
import os


import matplotlib.pyplot as plt  # for graphing our mean rewards over time
import matplotlib.animation as animation

d = {TREASURE_CHAR: '1',  # blueish color
     PLAYER_CHAR: '2',  # green
     DRAGON_CHAR: '3',# red
     OBSTACLE_CHAR: '4'}

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

fig = plt.figure(figsize=(10,10))
for ep in range(TOT_EPISODES):
  #recreate the environment
    bilbo=QLearningAgent(PLAYER_CHAR)
    mondo=World(WORLD_DIM,bilbo=bilbo,obstacle=True)
    #print(World)
    #do Q-stuff
    #print(bilbo.get_pos())
    game_ended=False
    epoch = 0
    anim = []
    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
      epoch += 1
      epsilon_fear = bilbo.fear(epsilon)
      action = bilbo.get_action(epsilon,q_table,possible_moves)
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


#    if ep % 10 == 0:
      #print("epoch ", epoch)
      #print(mondo)
      #print(bilbo.get_current_state(),bilbo.treasure_gone(),bilbo.game_ended())
      #print("epoch used: ",epoch, " ep:", ep)
      #print(bilbo.reward())
      #os.system( 'clear' )

      if ep == TOT_EPISODES-1: #LAST EPISODE
          env = np.zeros((WORLD_DIM, WORLD_DIM), dtype=np.uint8)  # starts an rbg of our size
          if mondo.get_position(TREASURE_CHAR):
            env[WORLD_DIM - 1  - mondo.get_position(TREASURE_CHAR)[0],mondo.get_position(TREASURE_CHAR)[1]] = d[TREASURE_CHAR]  # sets the treasure location tile
          if mondo.get_position(PLAYER_CHAR):
            env[WORLD_DIM - 1 - mondo.get_position(PLAYER_CHAR)[0],mondo.get_position(PLAYER_CHAR)[1]] = d[PLAYER_CHAR]
          env[WORLD_DIM - 1 - mondo.get_position(DRAGON_CHAR)[0],mondo.get_position(DRAGON_CHAR)[1]] = d[DRAGON_CHAR]
          obstacles = np.argwhere(mondo.world==OBSTACLE_CHAR)
          for coord in obstacles:
                  env[WORLD_DIM - 1 - coord[0]][coord[1]]=d[OBSTACLE_CHAR]
          anim.append((plt.pcolor(env),))



    epsilon *= decay_epsilon
    print(ep)

im_ani = animation.ArtistAnimation(fig, anim, interval=50, repeat_delay=None,#put = 0 if you want to repeat
                                   blit=True)

plt.show()
print("Atlast the episilon value was ", epsilon)
#print(mondo)
