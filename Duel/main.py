import agents
import constants
import gui
import world


p = [agents.Agent(constants.PLAYER1_CHAR,
                  qfile=constants.FILE1),
     agents.Agent(constants.ENEMY1_CHAR,
                  qfile=constants.FILE2)]
world1 = world.World(p, False)
gui.GUI(world1)
