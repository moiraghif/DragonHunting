from CreateBilboWorld import *
import numpy as np
from agents import *
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

d = {TREASURE_CHAR: '16',
     PLAYER_CHAR: '5',
     DRAGON_CHAR: '10',
     OBSTACLE_CHAR: '20'}


TOT_EPISODES = 1
MAX_EPOCH = 1000

possible_moves = {'up':0, 'down':1, 'left':2, 'right':3}
inverse_possible_moves = {0: 'up',1: 'down',2: 'left',3: 'right'}


bilbo = DeepQLearningAgent(PLAYER_CHAR)
win = 0
lost = 0
nothingness = 0
tot_reward = 0
fig = plt.figure(figsize=(20,20))
for ep in range(TOT_EPISODES):
    anim =[]

    mondo=World(WORLD_DIM, bilbo=bilbo, obstacle=False, random_spawn=True)
    #do deep Q-stuff
    game_ended=False
    epoch = 0
    current_state=bilbo.get_state()
    env = mondo.create_env(d)

    anim.append((plt.pcolormesh(env, cmap='CMRmap'),))
    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
        epoch += 1
        mondo.move_dragon()
        action = bilbo.get_action(0, possible_moves)
        bilbo.move(inverse_possible_moves[action])()
        new_state = bilbo.get_state()
        reward = bilbo.reward(current_state, new_state)

        tot_reward += reward
        game_ended = bilbo.game_ended()
        current_state = new_state

        if reward == TREASURE_REWARD:
            win += 1
        elif reward == -DRAGON_PENALTY:
            lost += 1
        env = mondo.create_env(d)
        anim.append((plt.pcolormesh(env, cmap='CMRmap'),))
        if game_ended or epoch + 1 == MAX_EPOCH:
            plt.text(0, 0.5, "Total Reward:" + str(tot_reward) + " Total Epoch:" + str(epoch+1), color='white', fontsize=20)

print("Tot Won: {}, Tot Lost: {}, Tot Nothingness: {}, Tot Reward: {}, Epoch survived: {}".format(win, lost, nothingness, tot_reward, epoch))
im_ani = animation.ArtistAnimation(fig, anim, interval=30, repeat_delay=0,
                                   blit=False)

ax = plt.gca()
plt.axis('off')

writer = animation.FFMpegWriter(fps=30)
im_ani.save('./videos/animation_video_deep_15x15.mp4',writer=writer)


plt.show()
