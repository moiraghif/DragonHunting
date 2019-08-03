import agents
import world
import constants


MAX_EPOCHS = 500
EPISODES = 100000


def reset():
    players = [agents.Agent(constants.PLAYER1_CHAR,
                            qfile=constants.FILE1),
               agents.Agent(constants.ENEMY1_CHAR,
                            qfile=constants.FILE2)]
    world1 = world.World(players, True)
    world1.save_qtable()


def print_max_length(txt):
    max_l = len(str(txt))

    def print_txt(t):
        return " " * (max_l - len(str(t))) + str(t)

    return print_txt


fix_episode = print_max_length(EPISODES)
fix_epoch = print_max_length(MAX_EPOCHS)

for episode in range(0, EPISODES):
    players = [agents.Agent(constants.PLAYER1_CHAR,
                            qfile=constants.FILE1),
               agents.Agent(constants.ENEMY1_CHAR,
                            qfile=constants.FILE2)]
    world1 = world.World(players, False)
    print("Episode {}".format(fix_episode(episode)), end="\r")
    for epoch in range(MAX_EPOCHS):
        last_move = (epoch + 1) == MAX_EPOCHS
        for p in players:
            reward, done = p.get_action(last_move)
            if done:
                break
        if done:
            break
    world1.save_qtable()
