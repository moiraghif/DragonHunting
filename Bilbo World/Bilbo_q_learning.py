import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agents import QLearningAgent, WORLD_DIM
from CreateBilboWorld import *


d = {TREASURE_CHAR: '16',
     PLAYER_CHAR: '5',
     DRAGON_CHAR: '10',
     OBSTACLE_CHAR: '20'}


alpha = 0.5
gamma = 0.8
epsilon = 0.2
decay_epsilon = 0.99

#In case of 15x15
TOT_EPISODES=1000
MAX_EPOCH=1000

# In case of 20x20
#TOT_EPISODES=1_000
#MAX_EPOCH=3000

# In case of 25x25
#TOT_EPISODES=400
#MAX_EPOCH=1000

# In case of 30x30
#TOT_EPISODES=600
#MAX_EPOCH=1000
#epsilon = 0.3

# In case of 50x50
#TOT_EPISODES=1500
#MAX_EPOCH=1500
#epsilon = 0.4
#decay_epsilon = 0.994

#initalize the q_table:
possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}

file_name = "qtable_" + str(WORLD_DIM)

q_table=np.array([[[0.0 for moves in possible_moves]
           for x in range(WORLD_DIM)]
           for y in range(WORLD_DIM)]) \
           if not os.path.isfile(file_name + ".npy") \
           else np.load(file_name + ".npy")

fig = plt.figure(figsize=(20,20))
for ep in range(TOT_EPISODES):
  #recreate the environment
    bilbo=QLearningAgent(PLAYER_CHAR)
    mondo=World(WORLD_DIM,bilbo=bilbo,obstacle=True)
    np.random.seed()
    #print(World)
    #do Q-stuff
    #print(bilbo.get_pos())
    game_ended=False
    epoch = 0
    anim = []
    titles = []
    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
        epoch += 1
        epsilon_fear = bilbo.fear(epsilon)
        action = bilbo.get_action(epsilon,q_table,possible_moves)
        current_state = bilbo.get_current_state()
      #treasure_gone = bilbo.treasure_gone()

        old_q_val = q_table[current_state][action]
        bilbo.move(inverse_possible_moves[action])()

        new_state = bilbo.get_current_state()
      #treasure_gone = bilbo.treasure_gone()
        game_ended = bilbo.game_ended()
        reward = bilbo.reward()

        if reward == -DRAGON_PENALTY:
            new_q_val = reward
        elif epoch == MAX_EPOCH:
            reward = -TOO_MUCH_WALK_PENALTY
            new_q_val = -TOO_MUCH_WALK_PENALTY
        elif new_state==current_state:
            #any kind of obtacle which made bilbo not move
            new_q_val = -OBSTACLE_PENALTY #-10 in case of obstacle
        else:
            next_q_val = np.max(q_table[new_state])
            new_q_val = bilbo.learning_function(alpha,gamma,old_q_val,reward,next_q_val)

        q_table[current_state][action] = new_q_val
      #import ipdb; ipdb.set_trace()

    epsilon *= decay_epsilon
    print("episode:{0:5}, epoch used:{1:4} [{2:10}]{3:3}%".format(ep,epoch,'#'*(round(ep*10/TOT_EPISODES)+1),round((ep+1)*100/TOT_EPISODES)),end='\r')

print('')
np.save(file_name,q_table)


#Should be testing phase create a new file which is better!!

#bilbo=QLearningAgent(PLAYER_CHAR)
#mondo=World(WORLD_DIM,bilbo=bilbo,obstacle=True)
#print(World)
#do Q-stuff
#print(bilbo.get_pos())
#game_ended=False
#epoch = 0
#anim = []
#rewards = 0

#The First frame
#env = mondo.create_env(d)
#anim.append((plt.pcolormesh(env,cmap='CMRmap'),))

#while not game_ended and epoch < MAX_EPOCH:
#  epoch += 1
#  action = bilbo.get_action(0,q_table,possible_moves)
#  bilbo.move(inverse_possible_moves[action])()
#  game_ended = bilbo.game_ended()
#  reward = bilbo.reward()
#  rewards = rewards + reward

#  env = mondo.create_env(d)
#  anim.append((plt.pcolormesh(env,cmap='CMRmap'),))

#title = "Epoch: " + str(epoch) + ", Total Reward: " + str(rewards) + ", taining episodes: " + str(TOT_EPISODES)
#print(title)
#print("Last state of bilbo: ", mondo.get_state(),mondo.treasure_gone())
#
#im_ani = animation.ArtistAnimation(fig, anim, interval=30, repeat_delay=1000,
#                                   blit=True)
#writer = animation.FFMpegWriter(fps=30)

#print("Writing video on your FS")
#im_ani.save('animation_video_50x50.mp4',writer=writer)
#ax = plt.gca()
#ax.invert_yaxis()
#plt.axis('off')
#plt.title(title)
#plt.show()
