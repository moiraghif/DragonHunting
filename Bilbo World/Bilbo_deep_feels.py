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

TOT_EPISODES=50000
MAX_EPOCH = 800

possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}


#alpha = 0.5
gamma = 0.8
epsilon = 0.5
epsilon_min = 0.01
decay_epsilon = 0.9999

bilbo=DeepQLearningAgentImage(PLAYER_CHAR)
won = 0
lost = 0
for ep in range(TOT_EPISODES):
    #recreate the environment
    mondo=World(WORLD_DIM,bilbo=bilbo,obstacle=True)
    #do deep Q-stuff
    np.random.seed()
    game_ended=False
    epoch = 0
    current_state=bilbo.get_state()
    initial = mondo.get_position(PLAYER_CHAR)
    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
        epoch += 1
        #ipdb.set_trace()
        epsilon_fear = bilbo.fear(epsilon) if epsilon > epsilon_min else bilbo.fear(epsilon_min)
        action = bilbo.get_action(epsilon_fear,possible_moves)
        #treasure_gone = bilbo.treasure_gone()
        bilbo.move(inverse_possible_moves[action])()
        reward = bilbo.reward()

        new_state = bilbo.get_state()
        #treasure_gone = bilbo.treasure_gone()
        game_ended = bilbo.game_ended()

        if reward==TREASURE_REWARD:
            won+=1
        if reward==-DRAGON_PENALTY:
            lost+=1
        #reward = bilbo.reward()
        bilbo.add_knowledge((current_state,action,reward,new_state,game_ended,epoch))
        current_state = new_state #Lol avevo dimenticato questo e non capivo perché preferisse sbattare contro i muri
        bilbo.train(gamma,MAX_EPOCH)
        #print("ep: ",ep ," epoch: ", epoch)

    ep += 1
    if ep % 1 == 0:
        #print("epoch ", epoch)
        print(mondo)
        print("Total Wins: ",won," Total Lost: ",lost, "Nothingness: ", ep-(won+lost))
        print("epoch used: ",epoch, " ep: ", ep, " epsilon: ",round(epsilon,4))
        print(bilbo.reward())
        #os.system( 'clear' )
    epsilon *= decay_epsilon
    #each 10 episode save the model
    if ep % 10 == 0:
        save_model(bilbo.q_nn,'deep_model.model')

#save the model again in case the episodes were not divisible by 10
save_model(bilbo.q_nn,'deep_model.model')
#print(q_table)
print("Atlast the episilon value was ", epsilon)
#print(mondo)

#np.save('deep_learning_weights',bilbo.q_nn.get_weights())

#testing
fig = plt.figure(figsize=(20,20))
anim =[]
mondo=World(WORLD_DIM,bilbo=bilbo,obstacle=True)
#do deep Q-stuff
game_ended=False
epoch = 0
#current_state=bilbo.get_state()
env = mondo.create_env(d)
anim.append((plt.pcolormesh(env,cmap='CMRmap'),))
while not game_ended and epoch < MAX_EPOCH:
  #the near it gets to the dragon the more random the movement
    epoch += 1
    #ipdb.set_trace()
    action = bilbo.get_action(-1,possible_moves)
    #treasure_gone = bilbo.treasure_gone()
    reward = bilbo.reward()
    bilbo.move(inverse_possible_moves[action])()

    #new_state = bilbo.get_state()
    #treasure_gone = bilbo.treasure_gone()
    game_ended = bilbo.game_ended()
    env = mondo.create_env(d)
    anim.append((plt.pcolormesh(env,cmap='CMRmap'),))
    print(mondo)


im_ani = animation.ArtistAnimation(fig, anim, interval=30, repeat_delay=1000,
                                   blit=True)
writer = animation.FFMpegWriter(fps=30)
#im_ani.save('animation_video_50x50.mp4',writer=writer)

ax = plt.gca()
plt.axis('off')
#plt.title(title)
plt.show()
