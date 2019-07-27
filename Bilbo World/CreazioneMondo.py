import numpy as np
import re
# import ipdb


DRAGON_CHAR = "D"
KNIGHT_CHAR = "K"
ARCHER_CHAR = "A"
OBSTACLE_CHAR = "█"
OBSTACLE_CONST = 0.12


class Agent:
    def __init__(self, char, item, vel):
        "Create a new agent"
        self.char = char
        self.world = None
        self.item = item
        self.velocity = vel
        # a list of possible movements that the agent can do
        movements = [self.move(d, l)
                     for l in range(1, self.velocity)
                     for d in ["up", "down", "left", "right"]]
        # a list of actions (functions) that the Agent can do
        self.actions = movements + []

    def initialize_world(self, world):
        self.world = world

    def move(self, d, l):
        "Generate the function to mode the agent"
        # yes, it is a clojure: it returns a function
        x = 0 if d not in ["right", "left"] \
            else l * (+1 if d == "right" else -1)
        y = 0 if d not in ["up", "down"] \
            else l * (+1 if d == "down" else -1)

        def make_movment():
            self.world.move_of(self, y=y, x=x)

        return make_movment

    def look_around(self):
        "Just look around"
        return self.world.explore()

    def __str__(self):
        return self.char

    def __eq__(self, other):
        "Check if two agents are the same"
        try:
            return self.char == other.char
        except NameError:
            return False


class World:
    def __init__(self, dragon, players, dim_x=15, dim_y=15):
        "Create a new World"
        self.dragon = dragon
        self.players = players
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.world = np.array([["" for x in range(self.dim_x)]
                               for y in range(self.dim_y)])
        self.insert_obstacles()
        self.world[0, round(self.dim_x / 2)] = str(dragon)
        for i, p in enumerate(players):
            self.world[self.dim_x - 1 - i,
                       self.dim_y - 1] = p
            p.initialize_world(self)
        dragon.initialize_world(self)

    def insert_obstacles(self):
        "Insert obstacles in the new world"
        for i in range(self.dim_x):
            for j in range(self.dim_y):
                if np.random.rand() > 1 - OBSTACLE_CONST:
                    self.world[i, j] = OBSTACLE_CHAR

    def get_agents(self):
        "Get all living agents in the world"
        return self.world[np.where(self.world and
                                   self.world != OBSTACLE_CHAR)]

    def end_game(self):
        "Is the game finished? In that case, return the winner(s)"
        agents = self.get_agents()
        if (agents == DRAGON_CHAR).any() or \
           np.in1d(agents, np.array([KNIGHT_CHAR, ARCHER_CHAR])).any():
            return list(agents)
        return None

    def is_free(self, pos):
        """Check if the cell is free:
        return False if not, the element otherwise"""
        # World.is_free((x, y))
        # check if cell is in the world
        if pos[0] < 0 or pos[0] > self.dim_x or \
           pos[1] < 0 or pos[1] > self.dim_y:
            return False
        return self.world[pos] == ""

    def move(self, pos_from, pos_to):
        "Move an element if possible"
        # if the initial position is free or
        # the finishing position is obstruited
        if not self.world[pos_from] or not self.is_free(pos_to):
            return False
        # move the agent
        self.world[pos_from], self.world[pos_to] = "", self.world[pos_from]
        return True

    def get_position(self, agent):
        "Get the position of an agent"
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                if self.world[y, x] == agent.char:
                    return (y, x)
        return None

    def move_of(self, agent, x=0, y=0):
        "Move the agent if possible"
        pos_from = self.get_position(agent)
        #ipdb.set_trace()
        # check if there is the agent on the board
        if not pos_from:
            return False
        # check if the movement is in just one direction and not null
        if (x > 0 and y > 0) or x + y == 0:
            return False
        # check if there are obstacles between
        #if (self.world[pos_from[0]:(pos_from[0] + y + 1),
        #               pos_from[1]:(pos_from[1] + x + 1)
        #               ] != "" ).any():
        #the vector above contains also the agent itselt in pos (pos_from[0],pos_from[1])
        #thus we need to give clearance even in case it has both '' and agent.__str__()
        if (not np.in1d(self.world[pos_from[0]:(pos_from[0] + y + 1),
                                   pos_from[1]:(pos_from[1] + x + 1)],
                        ["",agent.__str__()]).all()):
            return False
        # move the agent
        pos_to = (pos_from[0] + y, pos_from[1] + x)
        return self.move(pos_from, pos_to)

    def explore(self, agent, vr):
        "Return what an agent sees"
        return self.world

    def __str__(self):
        "Convert the world into string"
        txt = "┌" + "─" * (2 * self.dim_x - 1) + "┐\n│"
        txt += "│\n│".join(map(lambda r: " ".join([str(i) if i else " "
                                                   for i in r]),
                               self.world))
        txt += "│\n" + "└" + "─" * (2 * self.dim_x - 1) + "┘"
        return re.sub(OBSTACLE_CHAR + r"\s" + OBSTACLE_CHAR,
                      3 * OBSTACLE_CHAR, txt)


if __name__ == "__main__":
    players = [Agent(ARCHER_CHAR, "archer", 2),
               Agent(KNIGHT_CHAR, "sword", 1)]
    dragon = Agent(DRAGON_CHAR, "fire", 3)
    world1 = World(dragon, players)
    print(world1)
