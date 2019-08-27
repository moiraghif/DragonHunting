from CreateBilboWorld import *
from agents import *
from random import choice, randint
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

d = {TREASURE_CHAR: '16',
     PLAYER_CHAR: '5',
     DRAGON_CHAR: '10',
     OBSTACLE_CHAR: '20'}

memory = deque(maxlen=5000)
hight_reward_memory = deque(maxlen=5000)

MAX_EPOCH = 150
TOT_GENERATION = 300
ELITISM = 0.1
MUTATION = 0.05
n_pop = 100

possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}

gamma = 0.8
mondo=World(WORLD_DIM, bilbo=None, obstacle=False, random_spawn=True)

def train(bilbo, i):
    if len(memory) < MIN_MEMORY or len(hight_reward_memory) < MIN_MEMORY:
        return bilbo

    start_time = time.time()
    minibatch = random.sample(memory, 16)
    minibatch.extend(random.sample(hight_reward_memory,16))

    current_states = np.array([memory[0] for memory in minibatch])
    current_qs_list = bilbo.q_nn.predict(current_states)

    next_states = np.array([memory[3] for memory in minibatch])
    future_qs_list = bilbo.q_nn.predict(next_states)

    X = []
    y = []

        # Now we need to enumerate our batches
    for index, (current_state, action, reward, next_current_state, game_ended) in enumerate(minibatch):
        # almost like with Q Learning, but we use just part of equation here
        if game_ended or reward == -OBSTACLE_PENALTY or reward == TREASURE_REWARD:
            new_q = reward
        else:
            max_future_q = np.max(future_qs_list[index])
            new_q = reward + gamma * max_future_q

        # Aggiorna i q-value
        current_qs = current_qs_list[index]
        current_qs[action] = new_q

        X.append(current_state)
        y.append(current_qs)

    bilbo.q_nn.fit(np.array(X), np.array(y), batch_size=32, verbose=0, shuffle=False)
    print("The {0:3} training took {1:2}s".format(i,round(time.time()-start_time)), end='\r')
    return bilbo

def recorded_gameplay(bilbo):
    fig = plt.figure()
    anim =[]
    mondo.set_bilbo(bilbo)
    #do deep Q-stuff
    game_ended = False
    epoch = 0
    current_state=bilbo.get_state()
    env = mondo.create_env(d)

    anim.append((plt.pcolormesh(env, cmap='CMRmap'),))
    while not game_ended and epoch < MAX_EPOCH:
      #the near it gets to the dragon the more random the movement
        epoch += 1
        action = bilbo.get_action()
        bilbo.move(inverse_possible_moves[action])()
        new_state = bilbo.get_state()
        reward = bilbo.reward(current_state, new_state)

        game_ended = bilbo.game_ended()
        current_state = new_state
        env = mondo.create_env(d)
        anim.append((plt.pcolormesh(env, cmap='CMRmap'),))
        if reward in [-DRAGON_PENALTY, -OBSTACLE_PENALTY]:
            break
    im_ani = animation.ArtistAnimation(fig, anim, interval=30, repeat_delay=0,
                                       blit=False)

    ax = plt.gca()
    plt.axis('off')
    plt.close()

    return im_ani

def play_game(bilbo, n_times=1, max_epoch=MAX_EPOCH):
    #n_times=1
    to_print = False
    tot_reward = 0
    for ep in range(n_times):
        #mondo = World(WORLD_DIM, bilbo=bilbo, obstacle=False, random_spawn=True)
        mondo.set_bilbo(bilbo)
        #do deep Q-stuff
        np.random.seed(None)
        game_ended = False
        epoch = 0
        current_state = bilbo.get_state()
        while not game_ended and epoch < max_epoch:
            epoch += 1
            #mondo.move_dragon()
            action = bilbo.get_action()

            bilbo.move(inverse_possible_moves[action])()
            new_state = bilbo.get_state()
            reward = bilbo.reward(current_state, new_state)
            tot_reward += reward
            knowledge = (current_state,action,reward,new_state,reward in [-DRAGON_PENALTY,-OBSTACLE_PENALTY])
            current_state = new_state
            if reward in [-DRAGON_PENALTY, TREASURE_REWARD]:
                hight_reward_memory.append(knowledge)
            else:
                memory.append(knowledge)
            if to_print:
                print(mondo)
            if reward in [-DRAGON_PENALTY, -OBSTACLE_PENALTY]:
                break
    return tot_reward/n_times #average reward

def pop_fitness(pop, max_epoch=MAX_EPOCH):
    # calculating the fitness value by playing a game with the given weights in chromosome
    fitness = []
    for i in range(len(pop)):
    #Parallel(n_jobs=-1)(delayed(play_game)(bilbo) for bilbo in pop)
        sub_fitness = play_game(pop[i], 1, max_epoch)
        print('fitness value of chromosome '+ str(i+1) +' :  ', round(sub_fitness,2), end='\r')
        fitness.append(sub_fitness)
    return np.array(fitness)

def mutate(element_agent, mutationRate=0.1):
    mutated = []
    element = element_agent.get_weights()
    for array in element:
        sub_mutated = []
        for elem in array:
            if np.random.uniform(0, 1) < mutationRate:
                extra = np.random.uniform(-2, 2)
                sub_mutated.append(elem*extra)
            else:
                sub_mutated.append(elem)
        mutated.append(np.array(sub_mutated))
    element_agent.set_weights(mutated)
    return element_agent

def crossover(mother_agent, father_agent, old_agent):
    mother = mother_agent.get_weights()
    father = father_agent.get_weights()
    child = []
    for i in range(len(mother)):
        sub_child = []
        for j in range(len(mother[i])):
            sub_child.append(random.choice([mother[i][j], father[i][j]]))
        child.append(np.array(sub_child))
    old_agent.set_weights(child)
    return old_agent

def parents(pop, elitism):
    fitness = pop_fitness(pop)
    fitness_sorted = np.argsort(fitness)[::-1] #invertire
    n_pop = len(pop)
    #parent_size = round(n_pop*elitism)
    parents = []
    for i in range(len(pop)):
        parents.append(pop[fitness_sorted[i]])
    best_model = pop[fitness_sorted[0]]
    return parents, round(np.mean(fitness),3), round(np.max(fitness),3), best_model

def _new_pop(pop, elitism=0.1, mutationRate=0.1):
    start_time = time.time()
    children = []
    eliteSize = round(len(pop)*elitism)
    length = len(pop) - eliteSize
    pop_ordered, average_fitness, max_fitness, best_model = parents(pop, elitism=elitism)
    print("calculated parents in:{}s".format(round(time.time()-start_time)))
    parent_pool = pop_ordered[:eliteSize]
    for i in range(0, eliteSize):
        children.append(train(parent_pool[i], i))

    for i in range(0, length):
        sample = random.sample(parent_pool,2)#per evitare madre==padre
        mother = sample[0]
        father = sample[1]
        old = pop_ordered[i + eliteSize]
        child = crossover(mother, father, old)
        children.append(child)
    for i in range(len(children)):
        children[i] = mutate(children[i], mutationRate)
    return children, average_fitness, max_fitness, best_model

pop = [DeepQLearningAgentGA(PLAYER_CHAR) for i in range(n_pop)]

pop_fit = pop_fitness(pop)
average_fit = round(np.mean(pop_fit),3)
average_reward = [average_fit]
max_fit = round(np.max(pop_fit),3)
max_reward = [max_fit]

start_time = time.time()
for gen in range(TOT_GENERATION):
    pop, average_fit, max_fit, best_model = _new_pop(pop, ELITISM, MUTATION)
    print("New gen average:{} New gen Max:{}, Generation:{}, Time:{}s".format(average_fit, max_fit, gen + 1, round(time.time() - start_time)))
    average_reward.append(average_fit)
    max_reward.append(max_fit)

best_model.save_model()
