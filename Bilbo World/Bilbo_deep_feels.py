import matplotlib.pyplot as plt
import matplotlib.animation as animation
from CreateBilboWorld import *
from agents import *

d = {TREASURE_CHAR: '16',
     PLAYER_CHAR: '5',
     DRAGON_CHAR: '10',
     OBSTACLE_CHAR: '20'}

TOT_EPISODES = 10000
MAX_EPOCH = 400

possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}


gamma = 0.8
epsilon = 0.5
epsilon_min = 0.01
decay_epsilon = 0.9995

bilbo = DeepQLearningAgentImage(PLAYER_CHAR)
won = 0
lost = 0
for ep in range(TOT_EPISODES):
    #recreate the environment
    mondo = World(WORLD_DIM,bilbo=bilbo,obstacle=True)
    #do deep Q-stuff
    np.random.seed()
    game_ended = False
    epoch = 0
    current_state = bilbo.get_state()
    initial = mondo.get_position(PLAYER_CHAR)
    while not game_ended and epoch < MAX_EPOCH:
        #the near it gets to the dragon the more random the movement
        #in order to explore more around it
        epoch += 1
        epsilon_fear = bilbo.fear(epsilon) if epsilon > epsilon_min else bilbo.fear(epsilon_min)
        action = bilbo.get_action(epsilon_fear, possible_moves)

        bilbo.move(inverse_possible_moves[action])()
        reward = bilbo.reward()

        new_state = bilbo.get_state()
        game_ended = bilbo.game_ended()

        if reward == TREASURE_REWARD:
            won += 1
        if reward == -DRAGON_PENALTY:
            lost += 1

        bilbo.add_knowledge((current_state, action, reward, new_state, game_ended))
        current_state = new_state
        bilbo.train(gamma)


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

        save_model(bilbo.q_nn,'deep_model_'+str(WORLD_DIM)+'.model')

#save the model again in case the MAX_episodes was not divisible by 10
save_model(bilbo.q_nn,'deep_model_'+str(WORLD_DIM)+'.model')
print("Atlast the episilon value was ", epsilon)


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
