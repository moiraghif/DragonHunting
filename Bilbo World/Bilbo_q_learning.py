import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agents import QLearningAgent, WORLD_DIM
from CreateBilboWorld import *
from renderer import render_world


d = {TREASURE_CHAR: '16',
     PLAYER_CHAR: '5',
     DRAGON_CHAR: '10',
     OBSTACLE_CHAR: '20'}


alpha = 0.5
gamma = 0.8
epsilon = 0.2
decay_epsilon = 0.99
MIN_EPSILON = 0.01


#In case of 15x15
TOT_EPISODES = 1001
MAX_EPOCH = 1000

# In case of 20x20
#TOT_EPISODES = 10000
#MAX_EPOCH = 3000

# In case of 25x25
#TOT_EPISODES = 15000
#MAX_EPOCH = 1000

# In case of 30x30
#TOT_EPISODES = 20000
#MAX_EPOCH = 1000
#epsilon = 0.3

# In case of 50x50
#TOT_EPISODES = 25000
#MAX_EPOCH = 1500
#epsilon = 0.4
#decay_epsilon = 0.994

#initalize the q_table:
possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}

file_name = "./models/qtable_" + str(WORLD_DIM)

q_table = np.array([[[0.0 for moves in possible_moves]
                     for x in range(WORLD_DIM)]
                    for y in range(WORLD_DIM)]) \
           if not os.path.isfile(file_name + ".npy") \
           else np.load(file_name + ".npy")

fig = plt.figure(figsize=(20, 20))

epochs = []
rewards = []
epsilons = []
tot_wins = []
tot_loss = []
policy = []
win = 0
loss = 0

for ep in range(TOT_EPISODES):
  #recreate the environment
    bilbo = QLearningAgent(PLAYER_CHAR)
    mondo = World(WORLD_DIM, bilbo=bilbo, obstacle=True)
    np.random.seed()
    game_ended = False
    epoch = 0
    tot_reward = 0
    if ep % 10 == 0:
        a = plt.imshow(render_world(mondo.world,WORLD_DIM,q_table,ep), animated=True)
        policy.append((a,))
    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
        epoch += 1
        epsilon_fear = bilbo.fear(epsilon)
        action = bilbo.get_action(epsilon, q_table, possible_moves)
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
            loss += 1
        elif reward == TREASURE_REWARD:
            new_q_val = reward
            win += 1
        elif new_state == current_state:
            #any kind of obtacle which made bilbo not move
            new_q_val = -OBSTACLE_PENALTY #-10 in case of obstacle
        else:
            next_q_val = np.max(q_table[new_state])
            new_q_val = bilbo.learning_function(alpha, gamma, old_q_val, reward, next_q_val)

        q_table[current_state][action] = new_q_val
        tot_reward += reward

    epsilon = epsilon*decay_epsilon if epsilon > MIN_EPSILON else MIN_EPSILON
    epochs.append(epoch)
    rewards.append(tot_reward)
    epsilons.append(epsilon)
    tot_wins.append(win)
    tot_loss.append(loss)

    print("episode:{0:5}, epoch used:{1:4} [{2:10}]{3:3}%".format(ep,epoch,'#'*(round(ep*10/TOT_EPISODES)+1),round((ep+1)*100/TOT_EPISODES)),end='\r')

print('')
np.save(file_name, q_table)

#policy video
ani = animation.ArtistAnimation(fig, policy, interval=1, blit=True, repeat_delay=1000)
writer = animation.FFMpegWriter(fps=25)
plt.axis('off')
ani.save('./videos/policy_video_'+str(WORLD_DIM)+'.mp4', writer=writer)
plt.axis('on')

plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)
def cumsum_sma(array, period):
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

#plot for Reward
fig = plt.figure(figsize=(20, 20))
reward, = plt.plot(rewards)
ra_reward, = plt.plot(cumsum_sma(rewards, 100))
plt.legend([reward, ra_reward], ['Reward', 'Rolling Average'], loc=5, prop={'size': 20})
plt.xlabel('Episode', fontsize=40)
plt.ylabel('Reward', fontsize=40)
fig.savefig('./plots/reward_'+str(WORLD_DIM)+'.png')


#Plot for epoch required
fig = plt.figure(figsize=(20, 20))
reward, = plt.plot(epochs)
ra_reward, = plt.plot(cumsum_sma(epochs, 100))
plt.legend([reward, ra_reward], ['Epochs', 'Rolling Average'], loc=5, prop={'size': 20})
plt.xlabel('Episodes', fontsize=40)
plt.ylabel('Epochs', fontsize=40)
fig.savefig('./plots/epoch_'+str(WORLD_DIM)+'.png')
