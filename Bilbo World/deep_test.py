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

TOT_EPISODES = 1000
MAX_EPOCH = 400

possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}


bilbo=DeepQLearningAgentImage(PLAYER_CHAR)
win=0
lost=0
nothingness=0
for ep in range(TOT_EPISODES):
    #fig = plt.figure(figsize=(20,20))
    anim =[]
    mondo=World(WORLD_DIM,bilbo=bilbo,obstacle=True)
    #do deep Q-stuff
    game_ended=False
    epoch = 0
    #current_state=bilbo.get_state()
    env = mondo.create_env(d)

    #anim.append((plt.pcolormesh(env,cmap='CMRmap'),))
    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
        epoch += 1
        #ipdb.set_trace()
        action = bilbo.get_action(-1,possible_moves)
        #treasure_gone = bilbo.treasure_gone()
        bilbo.move(inverse_possible_moves[action])()
        reward = bilbo.reward()

        #new_state = bilbo.get_state()
        #treasure_gone = bilbo.treasure_gone()
        game_ended = bilbo.game_ended()
        #current_state = new_state

        env = mondo.create_env(d)
        #anim.append((plt.pcolormesh(env,cmap='CMRmap'),))
    #print(mondo)
    if reward==TREASURE_REWARD:
        win+=1
    elif reward==-DRAGON_PENALTY:
        lost+=1
    else:
        nothingness+=1
    print("Tot Won: {}, Tot Lost: {}, Tot Nothingness: {}".format(win,lost,nothingness), end="\r")

print("Tot Won: {}, Tot Lost: {}, Tot Nothingness: {}, epoch {}".format(win,lost,nothingness,epoch))
#import ipdb; ipdb.set_trace()
#if game_ended:
#    im_ani = animation.ArtistAnimation(fig, anim, interval=30, repeat_delay=0,
#                                   blit=True)
#    writer = animation.FFMpegWriter(fps=30)
#im_ani.save('animation_video_50x50.mp4',writer=writer)

#    ax = plt.gca()
#    plt.axis('off')
#    #plt.title(title)
#    plt.show()
#else:
#    print("Ha fatto schifo questa volta")
