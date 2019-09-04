import time
import agents
import GridWorld
import constants


start_time = time.time()

MAX_EPOCHS = 200
EPISODES = 50000
EPISILON_DECAY = 0.9999
MIN_EPSILON = 0.01

epsilon = constants.EPSILON
players = [agents.Agent(constants.PLAYER1_CHAR),
           agents.Agent(constants.PLAYER2_CHAR),
           agents.Agent(constants.DRAGON_CHAR)]

world1 = GridWorld.World(players, True)
game_ended = 0
for ep in range(0, EPISODES):

    players = [agents.Agent(constants.PLAYER1_CHAR),
               agents.Agent(constants.PLAYER2_CHAR),
               agents.Agent(constants.DRAGON_CHAR)]

    world1 = GridWorld.World(players,False)
    players_alive = {p: True for p in players}
    for epoch in range(MAX_EPOCHS):
        for p in players:
            if not players_alive[p]:
                continue
            reward, done = p.get_action(epoch+1 == MAX_EPOCHS, epsilon)
            if done:
                break
        if done:
            #import ipdb; ipdb.set_trace()
            if epoch + 1 != MAX_EPOCHS:
                game_ended += 1
            break
    epsilon = epsilon*EPISILON_DECAY if epsilon > MIN_EPSILON else MIN_EPSILON
    time_used = time.time()-start_time
    if ep%10 == 0:
        world1.save_qtable()
    print("Episode:{0:5}, game ended:{1:4} Training Status: [{2:10}]{3:3}%, time: {4}s {5}".format(ep+1,game_ended,'#'*(round(ep*10/EPISODES)+1),round((ep+1)*100/EPISODES),round(time_used), epoch),end='\r')

print("Episode:{0:5}, epoch used:{1:4} Training Status: [{2:10}]{3:3}%, time: {4}s".format(ep,epoch+1,'#'*(round(ep*10/EPISODES)+1),round((ep+1)*100/EPISODES),round(time_used)))
