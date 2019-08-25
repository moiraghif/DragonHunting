import time
import agents
import GridWorld
import constants


start_time = time.time()

MAX_EPOCHS = 200
EPISODES = 5000
EPISILON_DECAY = 0.999
MIN_EPSILON = 0.01

epsilon = constants.EPSILON

players = [agents.DeepAgent(constants.PLAYER1_CHAR),
           agents.DeepAgent(constants.PLAYER2_CHAR),
           agents.DeepAgent(constants.DRAGON_CHAR)]

tot_rew = {p:[] for p in players}
tot_lost = {p:0 for p in players}

game_ended=0
for ep in range(0, EPISODES):
    rewards = {p:0 for p in players}
    for p in players:
        p.reset()
    world1 = GridWorld.World(players, False)
    players_alive = {p: True for p in players}
    for epoch in range(MAX_EPOCHS):
        for p in players:
            if not players_alive[p]:
                continue
            current_state, fn, reward, next_state, done = p.get_action(epoch+1 == MAX_EPOCHS,epsilon)
            rewards[p] += reward

            if reward >= 100 or reward <= -50:
                p.add_high_knowledge((current_state, fn, reward, next_state, done))
                p.train()
            if done:
                tot_lost[p] += 1
                break
            else:
                p.add_knowledge((current_state, fn, reward, next_state, done))
                p.train()
        if done or epoch + 1 == MAX_EPOCHS:
            for p in players:
                tot_rew[p].append(rewards[p])
        if done:
            game_ended += 1
            break
    print(world1)
    epsilon = epsilon*EPISILON_DECAY if epsilon > MIN_EPSILON else MIN_EPSILON
    time_used = time.time()-start_time
    if ep%10 == 0:
        world1.save_NN()
    print("Episode:{0:5}, game ended:{1:4} Training Status: [{2:10}]{3:3}%, time: {4}s".format(ep+1,game_ended,'#'*(round(ep*10/EPISODES)+1),round((ep+1)*100/EPISODES),round(time_used)))
    print("P:{}:{}:{}, Q:{}:{}:{}, D:{}:{}:{}    {}".format(rewards[players[0]],players[0].healt,tot_lost[players[0]],
                                                      rewards[players[1]],players[1].healt,tot_lost[players[1]],
                                                      rewards[players[2]],players[2].healt,tot_lost[players[2]],
                                                      round(epsilon,3)))

#print("Episode:{0:5}, epoch used:{1:4} Training Status: [{2:10}]{3:3}%, time: {4}s".format(ep,epoch+1,'#'*(round(ep*10/EPISODES)+1),round((ep+1)*100/EPISODES),round(time_used)))
