import agents
import constants
import gui
import world


p = [agents.Agent(constants.PLAYER1_CHAR),
     agents.Agent(constants.PLAYER2_CHAR),
     agents.Agent(constants.ENEMY1_CHAR),
     agents.Agent(constants.ENEMY2_CHAR),
     agents.Agent(constants.DRAGON_CHAR)]
world1 = world.World(p, False)
gui.GUI(world1)
