import agents
import GridWorld
import constants
from time import sleep
import os

MAX_EPOCHS = 100

players = [agents.Agent(constants.PLAYER1_CHAR),
           agents.Agent(constants.PLAYER2_CHAR),
           agents.Agent(constants.DRAGON_CHAR)]

world1 = GridWorld.World(players,False)
players_alive = {p: True for p in players}
for epoch in range(MAX_EPOCHS):
    for p in players:
        if not players_alive[p]:
            continue
        reward, done = p.get_action(epoch+1 == MAX_EPOCHS,0)

        os.system('clear')
        print(world1)
        sleep(0.03)

        if done:
            break
    if done:
        break
