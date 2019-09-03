import time
import matplotlib.pyplot as plt
from CreateBilboWorld import *
from agents import *

def cumsum_average(array):
    ret = np.cumsum(array, dtype=float)
    ones = np.array(range(1, len(array)+1))
    return ret / ones

d = {TREASURE_CHAR: '16',
     PLAYER_CHAR: '5',
     DRAGON_CHAR: '10',
     OBSTACLE_CHAR: '20'}

TOT_EPISODES = 20000
MAX_EPOCH = 150

possible_moves = {'up':0, 'down':1, 'left':2, 'right':3}
inverse_possible_moves = {0:'up', 1:'down', 2:'left', 3:'right'}


gamma = 0.8
epsilon = 0.5
epsilon_min = 0.01
decay_epsilon = 0.9997

bilbo = DeepQLearningAgent(PLAYER_CHAR)

won = 0
lost = 0
epochs = []
rewards = []
epsilons = []
tot_wins = []
tot_loss = []
start_time = time.time()

fig = plt.figure(figsize=(15, 10))
plt.grid(True)
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
        mondo.move_dragon()
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
        if reward in [TREASURE_REWARD, -DRAGON_PENALTY]:
            bilbo.add_high_knowledge((current_state, action, reward, new_state, game_ended))
        else:
            bilbo.add_knowledge((current_state, action, reward, new_state, game_ended))
        current_state = new_state
        bilbo.train(gamma)
        if not reward in [-2, 1]:
            tot_reward += reward
        if end_game:
            break
        if tot_wins:
            if epoch == MAX_EPOCH and won == tot_wins[-1]:
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
        save_model(bilbo.q_nn, './models/deep_model_'+str(WORLD_DIM)+'.model')
        plt.subplot(221)
        reward_plot, = plt.plot(rewards, color='tab:blue')
        ra_reward, = plt.plot(cumsum_average(rewards), linewidth=5, color='tab:orange')
        plt.legend([reward_plot, ra_reward], ['Reward', 'Cumulative Average'], loc='best')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)

        plt.subplot(222)
        win_plot, = plt.plot(tot_wins, color='tab:blue')
        lost_plot, = plt.plot(tot_loss, color='tab:orange')
        plt.legend([win_plot, lost_plot], ['Total Wins', 'Total Non Wins'], loc='best')
        plt.xlabel('Episodes')
        plt.ylabel('Tot Won and Lost')
        plt.grid(True)
        plt.draw()
        plt.pause(0.001)

#save the model again in case the MAX_episodes was not divisible by 10
save_model(bilbo.q_nn, './models/deep_model_'+str(WORLD_DIM)+'.model')
plt.show()
### PLOTS
plt.rc('xtick', labelsize=15)    # fontsize of the tick labels
plt.rc('ytick', labelsize=15)

fig = plt.figure(figsize=(20, 20))
reward, = plt.plot(rewards, color='tab:blue')
ra_reward, = plt.plot(cumsum_average(rewards), linewidth=2, color='tab:orange')
plt.legend([reward, ra_reward], ['Reward', 'Cumulative Average'], loc='best', prop={'size': 20})
plt.xlabel('Episode', fontsize=40)
plt.ylabel('Reward', fontsize=40)
plt.grid(True)
fig.savefig('./plots/reward_deep_'+str(WORLD_DIM)+'.png')


fig = plt.figure(figsize=(20, 20))
win, = plt.plot(tot_wins)
lost, = plt.plot(tot_loss)
plt.legend([win, lost], ['Total Wins', 'Total Non Wins'], loc='best', prop={'size': 20})
plt.xlabel('Episodes', fontsize=40)
plt.ylabel('Tot Won and Lost', fontsize=40)
plt.grid(True)
fig.savefig('./plots/epoch_deep_'+str(WORLD_DIM)+'.png')
