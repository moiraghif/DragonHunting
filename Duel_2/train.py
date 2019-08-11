import agents
import world
import constants
import numpy as np


MAX_EPOCHS = 400
EPISODES = 10000
EPISILON_DECAY = 0.995
MIN_EPSILON = 0.01

epsilon = constants.EPSILON

def reset():
    players = [agents.Agent(constants.PLAYER1_CHAR),
               agents.Agent(constants.PLAYER2_CHAR),
               agents.Agent(constants.ENEMY1_CHAR),
               agents.Agent(constants.ENEMY2_CHAR),
               agents.Agent(constants.DRAGON_CHAR)]
    world1 = world.World(players, True)
    world1.save_qtable()


#reset()

def print_max_length(txt):
    max_l = len(str(txt))

    def print_txt(t):
        return " " * (max_l - len(str(t))) + str(t)

    return print_txt


fix_episode = print_max_length(EPISODES)
fix_epoch = print_max_length(MAX_EPOCHS)

for episode in range(0, EPISODES):
    players = [agents.Agent(constants.PLAYER1_CHAR),
               agents.Agent(constants.PLAYER2_CHAR),
               agents.Agent(constants.ENEMY1_CHAR),
               agents.Agent(constants.ENEMY2_CHAR),
               agents.Agent(constants.DRAGON_CHAR)]
    players_alive = {p: True for p in players}
    world1 = world.World(players, False)
    for epoch in range(MAX_EPOCHS):
        b_done = False
        last_move = (epoch + 1) == MAX_EPOCHS
        for p in players:
            if not players_alive[p]:
                continue
            reward, done = p.get_action(last_move,epsilon)

            if done:
                players_alive[p] = False
                b_done = np.sum(np.array([
                    v for k, v in players_alive.items()])) == 1
                if b_done:
                    break
        if b_done:
            break
    epsilon = epsilon*EPISILON_DECAY if epsilon > MIN_EPSILON else MIN_EPSILON
    print(world1)
    print("Episode {} , epsilon {}".format(fix_episode(episode),round(epsilon,4)))
    what_to_print = ""
    for p in players:
        health = '0'+str(p.healt) if p.healt < 10 else str(p.healt)
        what_to_print = what_to_print + p.char + ":" + health + "\t"
    print(what_to_print)

    world1.save_qtable()
