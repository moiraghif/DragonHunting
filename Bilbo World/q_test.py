import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from agents import QLearningAgent, WORLD_DIM
from CreateBilboWorld import *


d = {TREASURE_CHAR: '16',
     PLAYER_CHAR: '5',
     DRAGON_CHAR: '10',
     OBSTACLE_CHAR: '20'}

MAX_EPOCH = 100

possible_moves = {'up':0, 'down':1, 'left':2, 'right':3}
inverse_possible_moves = {0:'up', 1:'down', 2:'left', 3:'right'}

file_name = "./models/qtable_" + str(WORLD_DIM)


q_table=np.array([[[0.0 for moves in possible_moves]
           for x in range(WORLD_DIM)]
           for y in range(WORLD_DIM)]) \
           if not os.path.isfile(file_name + ".npy") \
           else np.load(file_name + ".npy")


fig = plt.figure(figsize=(20, 20))

bilbo = QLearningAgent(PLAYER_CHAR)
mondo = World(WORLD_DIM, bilbo=bilbo, obstacle=True)
game_ended = False
epoch = 0
anim = []
rewards = 0

env = mondo.create_env(d)
anim.append((plt.pcolormesh(env, cmap='CMRmap'),))

while not game_ended and epoch < MAX_EPOCH:
    epoch += 1
    action = bilbo.get_action(0, q_table, possible_moves)
    bilbo.move(inverse_possible_moves[action])()
    game_ended = bilbo.game_ended()
    reward = bilbo.reward()
    rewards = rewards + reward

    env = mondo.create_env(d)
    anim.append((plt.pcolormesh(env, cmap='CMRmap'),))

im_ani = animation.ArtistAnimation(fig, anim, interval=60, repeat_delay=1000,
                                   blit=False)

writer = animation.FFMpegWriter(fps=epoch)


print("Writing video on your FS")
im_ani.save('./videos/animation_video_15x15.mp4', writer=writer)
ax = plt.gca()
ax.invert_yaxis()
plt.axis('off')
plt.show()
