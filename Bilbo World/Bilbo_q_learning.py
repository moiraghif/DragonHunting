from CreateBilboWorld import *
import numpy as np
import ipdb
from agents import *
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

d = {TREASURE_CHAR: '16',
     PLAYER_CHAR: '5',
     DRAGON_CHAR: '10',
     OBSTACLE_CHAR: '20'}

#In case of 15x15
#TOT_EPISODES=800
#MAX_EPOCH=1_000

# In case of 20x20
#TOT_EPISODES=1_000
#MAX_EPOCH=3000

# In case of 25x25
TOT_EPISODES=5
MAX_EPOCH=800

#initalize the q_table:
possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}
q_table={}
for y in range(WORLD_DIM):
  for x in range(WORLD_DIM):
    for exist_reward in [0,1]:
        q_table[(y,x),exist_reward]=[0,0,0,0]


alpha = 0.5
gamma = 0.8
epsilon = 0.2
decay_epsilon = 0.99
rewards = []

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
      treasure_gone = bilbo.treasure_gone()

      old_q_val = q_table[current_state,treasure_gone][action]
      bilbo.move(inverse_possible_moves[action])()

      new_state = bilbo.get_current_state()
      treasure_gone = bilbo.treasure_gone()
      game_ended = bilbo.game_ended()
      reward = bilbo.reward()

      if reward == -DRAGON_PENALTY:
        new_q_val = reward
      elif epoch == MAX_EPOCH:
        reward = -TOO_MUCH_WALK_PENALTY
      elif new_state==current_state:
        #any kind of obtacle which made bilbo not move
        new_q_val = -OBSTACLE_PENALTY #-10 in case of obstacle
      else:
        next_q_val = np.max(q_table[new_state,treasure_gone])
        new_q_val = bilbo.learning_function(alpha,gamma,old_q_val,reward,next_q_val)

      q_table[current_state,treasure_gone][action] = new_q_val


    epsilon *= decay_epsilon
    print("episode: ", ep, " epoch used:", epoch, " final reward: ",
            reward ," epsilon: ", round(epsilon,4))

#ipdb.set_trace()

#testing_phase
bilbo=QLearningAgent(PLAYER_CHAR)
mondo=World(WORLD_DIM,bilbo=bilbo,obstacle=True)
#print(World)
#do Q-stuff
#print(bilbo.get_pos())
game_ended=False
epoch = 0
anim = []
rewards = 0

#The First frame
env = mondo.create_env(d)
anim.append((plt.pcolormesh(env,cmap='CMRmap'),))

while not game_ended and epoch < MAX_EPOCH:
  epoch += 1
  action = bilbo.get_action(0,q_table,possible_moves)
  #ipdb.set_trace()
  bilbo.move(inverse_possible_moves[action])()
  game_ended = bilbo.game_ended()
  reward = bilbo.reward()
  rewards = rewards + reward

  env = mondo.create_env(d)
  anim.append((plt.pcolormesh(env,cmap='CMRmap'),))



title = "Epoch: " + str(epoch) + ", Total Reward: " + str(rewards) + ", taining episodes: " + str(TOT_EPISODES)
print(title)
print("Last state of bilbo: ", mondo.get_state(),mondo.treasure_gone())

im_ani = animation.ArtistAnimation(fig, anim, interval=30, repeat_delay=1000,
                                   blit=True)
writer = animation.FFMpegWriter(fps=45)

print("Writing video on your FS")
#im_ani.save('animation_video.mp4',writer=writer)
ax = plt.gca()
ax.invert_yaxis()
plt.axis('off')
plt.title(title)
plt.show()