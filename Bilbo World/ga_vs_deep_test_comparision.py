from CreateBilboWorld import *
import numpy as np
from agents import *
import matplotlib.pyplot as plt

d = {TREASURE_CHAR: '16',
     PLAYER_CHAR: '5',
     DRAGON_CHAR: '10',
     OBSTACLE_CHAR: '20'}


TOT_EPISODES = 1000
MAX_EPOCH = 1000

possible_moves = {'up':0, 'down':1, 'left':2, 'right':3}
inverse_possible_moves = {0: 'up',1: 'down',2: 'left',3: 'right'}


bilbo = DeepQLearningAgent(PLAYER_CHAR)
lost = 0
nothingness = 0
rewards = []
for ep in range(TOT_EPISODES):
    anim = []
    win = 0
    tot_reward = 0
    mondo = World(WORLD_DIM, bilbo=bilbo, obstacle=False, random_spawn=True)
    #do deep Q-stuff
    game_ended = False
    epoch = 0
    current_state = bilbo.get_state()
    env = mondo.create_env(d)
    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
        epoch += 1
        mondo.move_dragon()
        action = bilbo.get_action(0, possible_moves)
        bilbo.move(inverse_possible_moves[action])()
        new_state = bilbo.get_state()
        reward = bilbo.reward(current_state, new_state)
        if not reward in [-2, 1]:
            tot_reward += reward
        game_ended = bilbo.game_ended()
        current_state = new_state

        if reward == TREASURE_REWARD:
            win += 1
        elif reward == -DRAGON_PENALTY:
            lost += 1
        env = mondo.create_env(d)
        if reward == -5:
            break
    rewards.append(tot_reward)
    print("ep: {} Tot Won: {}, Tot Lost: {}, Tot Nothingness: {}, Tot Reward: {}, Epoch survived: {}".format(ep + 1, win, lost, nothingness, tot_reward, epoch))

bilbo = DeepQLearningAgentGA(PLAYER_CHAR)
bilbo.q_nn = load_model('./models/deep_model_ga_'+str(WORLD_DIM)+'.model')
win = 0
rewards_ga = []
lost = 0
nothingness = 0
for ep in range(TOT_EPISODES):
    anim = []
    tot_reward = 0
    win = 0
    mondo = World(WORLD_DIM, bilbo=None, obstacle=False, random_spawn=True)
    mondo.set_bilbo(bilbo)
    #do deep Q-stuff
    game_ended = False
    epoch = 0
    current_state = bilbo.get_state()
    env = mondo.create_env(d)

    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
        epoch += 1
        mondo.move_dragon()
        action = bilbo.get_action()
        bilbo.move(inverse_possible_moves[action])()
        new_state = bilbo.get_state()
        reward = bilbo.reward(current_state, new_state)
        if not reward in [-2, 1]:
            tot_reward += reward
        game_ended = bilbo.game_ended()
        current_state = new_state

        if reward == TREASURE_REWARD:
            win += 1
        elif reward == -DRAGON_PENALTY:
            lost += 1
        env = mondo.create_env(d)
        if reward == -5:
            break

    rewards_ga.append(tot_reward)
    print("ep: {} Tot Won: {}, Tot Lost: {}, Tot Nothingness: {}, Tot Reward: {}, Epoch survived: {}".format(ep + 1, win, lost, nothingness, tot_reward, epoch))

plt.style.use('ggplot')
fig = plt.figure(figsize=(10, 10))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.boxplot([rewards,rewards_ga], notch=True, meanline=True, showmeans=True,
           meanprops=dict(linestyle='--', linewidth=2.5, color='tab:blue'),
           medianprops=dict(linestyle='--', linewidth=2.5, color='purple'),
           widths=(0.5, 0.5), labels=['Deep Q Learning','Deep Q Learning with GA'])#, color='tab:orange')
plt.text(0.78, np.mean(rewards) + 15, "Average: " + str(round(np.mean(rewards),2)), color='tab:blue', fontsize=15)
plt.text(1.8, np.mean(rewards_ga) + 15, "Average: " + str(round(np.mean(rewards_ga),2)), color='tab:blue', fontsize=15)
plt.ylabel('Rewards')
fig.savefig('./plots/reward_deep_comparision_'+str(WORLD_DIM)+'.png')
