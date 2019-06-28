import numpy as np
import re


DRAGON_CHAR = "D"
KNIGHT_CHAR = "K"
ARCHER_CHAR = "A"
OBSTABLE_CHAR = "█"


class Agent:
    def __init__(self, char, item, vr, vel):
        "Create a new agent"
        self.char = char
        self.world = None
        self.item = item
        self.view_range = vr
        self.velocity = vel
        # a list of possible movements that the agent can do
        self.movements = [self.move(d, l)
                          for l in range(self.velocity)
                          for d in ["up", "down", "left", "right"]]

    def initialize_world(self, world):
        self.world = world

    def move(self, d, l):
        "Generate the function to mode the agent"
        # yes, it is a clojure: it returns a function
        y = 0 if d not in ["right", "left"] \
            else l * (+1 if d == "right" else -1)
        x = 0 if d not in ["up" in "down"] \
            else l * (+1 if d == "down" else -1)

        def make_movment():
            self.world.move_of(self, y=y, x=x)
        return make_movment

    def look_around(self):
        "Just look around"
        return self.world.explore(self, self.view_range)

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
            self.world[self.dim_y - 1,
                       self.dim_x - 1 - i] = p
        for p in self.players:
            p.initialize_world(self)
        dragon.initialize_world(self)

    def insert_obstacles(self):
        "Insert obstacles in the new world"
        # TODO: generate new obstacles at random
        for i in range(2):
            for j in range(2):
                self.world[i, j] = OBSTABLE_CHAR

    def get_agents(self):
        "Get all living agents in the world"
        return self.world[np.where(self.world and
                                   self.world != OBSTABLE_CHAR)]

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
        return False if self.world[pos] == OBSTABLE_CHAR \
            else not self.world[pos]

    def move(self, pos_from, pos_to):
        "Move an element if possible"
        # if the initial position is free or
        # the finishing position is obstruited
        if not self.world[pos_from] or not self.is_free(pos_to):
            return False
        # move the agent
        # POSSIBLE ERROR: the second assignment must be a pointer
        self.world[pos_from], self.world[pos_to] = "", self.world[pos_from]
        return True

    def get_position(self, agent):
        "Get the position of an agent"
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                if self.world[y, x] == agent.char:
                    return (y, x)
        return None

    def move_of(self, agent, y=0, x=0):
        "Move the agent if possible"
        pos_from = self.get_position(agent)
        if not pos_from:
            return False
        pos_to = (pos_from[0] + x, pos_from[1] + y)
        return self.move(pos_from, pos_to)

    def explore(self, agent, vr):
        "Return what an agent see"
        pos = self.get_position(agent)
        y = (0 if pos[0] - vr < 0 else pos[0] - vr,
             self.dim_y - 1 if pos[0] + vr >= self.dim_y else pos[0] + vr)
        x = (0 if pos[1] - vr < 0 else pos[1] - vr,
             self.dim_x - 1 if pos[1] + vr >= self.dim_x else pos[1] + vr)
        return self.world[x[0]:x[1], y[0]:y[1]]

    def __str__(self):
        "Convert the world into string"
        txt = "┌" + "─" * (2 * self.dim_x - 1) + "┐\n│"
        txt += "│\n│".join(map(lambda r: " ".join([str(i) if i else " "
                                                   for i in r]),
                               self.world))
        txt += "│\n" + "└" + "─" * (2 * self.dim_x - 1) + "┘"
        return re.sub(OBSTABLE_CHAR + r"\s" + OBSTABLE_CHAR,
                      3 * OBSTABLE_CHAR, txt)


if __name__ == "__main__":
    players = [Agent(ARCHER_CHAR, "archer", 2, 2),
               Agent(KNIGHT_CHAR, "sword", 1, 1)]
    dragon = Agent(DRAGON_CHAR, "fire", 5, 3)
    world1 = World(dragon, players)
    print(world1)
