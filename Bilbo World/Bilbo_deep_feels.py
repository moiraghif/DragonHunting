import matplotlib.pyplot as plt
import matplotlib.animation as animation
from CreateBilboWorld import *
from agents import *
import time

d = {TREASURE_CHAR: '16',
     PLAYER_CHAR: '5',
     DRAGON_CHAR: '10',
     OBSTACLE_CHAR: '20'}

TOT_EPISODES = 10000
MAX_EPOCH = 500

possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}


gamma = 0.8
epsilon = 0.5
epsilon_min = 0.01
decay_epsilon = 0.999

bilbo = DeepQLearningAgentImage(PLAYER_CHAR)

won = 0
lost = 0
epochs = []
rewards = []
epsilons = []
tot_wins = []
tot_loss = []
start_time = time.time()
for ep in range(TOT_EPISODES):
    #recreate the environment
    mondo = World(WORLD_DIM, bilbo=bilbo, obstacle=False, random_spawn=True)
    #do deep Q-stuff
    np.random.seed()
    game_ended = False
    epoch = 0
    tot_reward = 0
    current_state = bilbo.get_state()
    initial = mondo.get_position(PLAYER_CHAR)
    end_game = False
    while not game_ended and epoch < MAX_EPOCH:
        #the near it gets to the dragon the more random the movement
        #in order to explore more around it
        epoch += 1
        #epsilon_fear = bilbo.fear(epsilon) if epsilon > epsilon_min else bilbo.fear(epsilon_min)
        action = bilbo.get_action(epsilon, possible_moves)

        bilbo.move(inverse_possible_moves[action])()
        new_state = bilbo.get_state()
        reward = bilbo.reward(current_state, new_state)

        if reward == TREASURE_REWARD:
            current_qs = bilbo.get_qs(current_state)
            current_qs[action] = reward
            current_qs = np.array([current_qs])
            bilbo.q_nn.fit(current_state.reshape(-1, bilbo.state_shape), current_qs, verbose=0)
            won += 1
        if reward == -DRAGON_PENALTY:
            end_game = True
            current_qs = bilbo.get_qs(current_state)
            current_qs[action] = reward
            current_qs = np.array([current_qs])
            bilbo.q_nn.fit(current_state.reshape(-1, bilbo.state_shape), current_qs, verbose=0)
            lost += 1
        if reward in [TREASURE_REWARD,-DRAGON_PENALTY]:
            bilbo.add_high_knowledge((current_state, action, reward, new_state, game_ended))
        else:
            bilbo.add_knowledge((current_state, action, reward, new_state, game_ended))
        current_state = new_state
        bilbo.train(gamma)
        tot_reward += reward
        if end_game:
            break
        if len(tot_wins)>0:
            if epoch == MAX_EPOCH and won==tot_wins[-1]:
                lost += 1


    ep += 1
    rewards.append(tot_reward)
    epsilons.append(epsilon)
    tot_wins.append(won)
    tot_loss.append(lost)

    print("Total Wins:{0} Total Lost:{1} episode:{2} [{3:10}]{4:3}% {5}s".format(won, lost, ep, '#'*(round(ep*10/TOT_EPISODES)+1),round((ep+1)*100/TOT_EPISODES), round(time.time() - start_time)), end='\r')
    epsilon = epsilon*decay_epsilon if epsilon > epsilon_min else epsilon_min
    #each 10 episode save the model
    if ep % 10 == 0:
        save_model(bilbo.q_nn,'./models/deep_model_'+str(WORLD_DIM)+'.model')

#save the model again in case the MAX_episodes was not divisible by 10
save_model(bilbo.q_nn,'deep_model_'+str(WORLD_DIM)+'.model')
print("Atlast the episilon value was ", epsilon)


### PLOTS
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)
def cumsum_sma(array, period):
    ret = np.cumsum(array, dtype=float)
    ret[period:] = ret[period:] - ret[:-period]
    return ret[period - 1:] / period

fig = plt.figure(figsize=(20,20))
reward, = plt.plot(rewards)
ra_reward, = plt.plot(cumsum_sma(rewards,100))
plt.legend([reward, ra_reward], ['Reward', 'Rolling Average'], loc = 5, prop={'size': 20})
plt.xlabel('Episode',fontsize=40)
plt.ylabel('Reward',fontsize=40)
fig.savefig('./plots/reward_deep_'+str(WORLD_DIM)+'.png')


fig = plt.figure(figsize=(20,20))
epochs, = plt.plot(epochs)
epochs_ra, = plt.plot(cumsum_sma(epochs,100))
plt.legend([reward, ra_reward], ['Epochs', 'Rolling Average'], loc = 5, prop={'size': 20})
plt.xlabel('Episodes',fontsize=40)
plt.ylabel('Epochs',fontsize=40)
fig.savefig('./plots/epoch_deep_'+str(WORLD_DIM)+'.png')
