import agents
import constants
import gui
import GridWorld


p = [agents.Agent(constants.PLAYER1_CHAR),
     agents.Agent(constants.PLAYER2_CHAR),
     agents.Agent(constants.DRAGON_CHAR)]
world1 = GridWorld.World(p, False)
gui.GUI(world1)
