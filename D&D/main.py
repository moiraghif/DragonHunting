import agents
import constants
import gui
import world


p = [agents.Agent(constants.PLAYER1_CHAR, "humans"),
     agents.Agent(constants.PLAYER2_CHAR, "humans"),
     agents.Agent(constants.ENEMY1_CHAR, "monsters"),
     agents.Agent(constants.ENEMY2_CHAR, "monsters")]
world1 = world.World(p)
gui.GUI(world1)
