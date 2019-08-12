import agents
import GridWorld
import constants

MAX_EPOCHS = 400

players = [agents.Agent(constants.PLAYER1_CHAR),
           agents.Agent(constants.PLAYER2_CHAR),
           agents.Agent(constants.DRAGON_CHAR)]

world1 = GridWorld.World(players,False)
players_alive = {p: True for p in players}
for epoch in range(MAX_EPOCHS):
    game_over = False
    for p in players:
        if not players_alive[p]:
            continue
        reward, done = p.get_action(epoch+1 == MAX_EPOCHS,0)
        print(world1)
        if done:
            players_alive[p] = False
            if p.char == 'D':
                game_over = True
        if p.char == 'D' and len(players_alive)==1:
            game_over=True
        if game_over:
            break
    if game_over:
        break
