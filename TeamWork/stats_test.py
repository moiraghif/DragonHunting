import time
import agents
import GridWorld
import constants


start_time = time.time()

MAX_EPOCHS = 200
EPISODES = 1000

players = [agents.Agent(constants.PLAYER1_CHAR),
           agents.Agent(constants.PLAYER2_CHAR),
           agents.Agent(constants.DRAGON_CHAR)]

game_ended = 0
tot_rew = {p.char:[] for p in players}
tot_lost = {p.char:0 for p in players}
for ep in range(0, EPISODES):
    who = 'N'
    players = [agents.Agent(constants.PLAYER1_CHAR),
               agents.Agent(constants.PLAYER2_CHAR),
               agents.Agent(constants.DRAGON_CHAR)]
    rewards = {p:0 for p in players}
    world1 = GridWorld.World(players,False)
    players_alive = {p: True for p in players}
    for epoch in range(MAX_EPOCHS):
        for p in players:
            if not players_alive[p]:
                continue
            reward, done = p.get_action(epoch+1 == MAX_EPOCHS, 0)
            rewards[p] += reward
            if done:
                if not p.alive():
                    who = p.char
                    tot_lost[p.char] += 1
                break
        if done:
            for p in players:
                tot_rew[p.char].append(rewards[p])
            if epoch + 1 != MAX_EPOCHS:
                game_ended += 1
            break
    time_used = time.time()-start_time
    print("Episode:{:5} lost by:{}, {}:{:5}  {}:{:5}  {}:{:5}".format(ep+1, who,
                                                                players[0].char,round(rewards[players[0]],1),
                                                                players[1].char,round(rewards[players[1]],1),
                                                                players[2].char,round(rewards[players[2]],1)))
print("{}:{}  {}:{}  {}:{}".format(players[0].char, tot_lost[players[0].char],
                                   players[1].char, tot_lost[players[1].char],
                                   players[2].char, tot_lost[players[2].char]))
