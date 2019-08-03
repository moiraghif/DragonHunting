import constants
import numpy as np


class World:
    def __init__(self, players, new_qtable):
        self.world = constants.WORLD.copy()
        self.dim_y, self.dim_x = self.world.shape
        self.players = {p: self.get_position(p) for p in players}
        for p in players:
            self.casual_spawn(p)
            p.initialize_world(self, new_qtable)
            # self.players[p] = self.get_position(p)

    def casual_spawn(self, agent):
        x = np.random.randint(self.dim_x)
        y = np.random.randint(self.dim_y)
        if self.is_free((y, x)):
            self.world[y, x] = agent.char
            self.players[agent] = np.array([y, x])
            return True
        return self.casual_spawn(agent)

    def is_free(self, pos):
        if any(np.array(pos) >= [self.dim_y, self.dim_x]) or \
           any(np.array(pos) < 0):
            return False
        return len(self.world[pos]) == 0

    def get_position(self, agent):
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                if self.world[y, x] == agent.char:
                    return np.array([y, x])
        return False

    def move_of(self, agent, x=0, y=0):
        delta = np.array([y, x])
        pos0 = self.players[agent]
        pos = tuple(pos0 + delta)
        delta0 = np.sum(abs(self.players[agent] -
                            self.players[self.get_enemy(agent)]))
        if self.is_free(pos):
            self.world[pos] = agent.char
            self.world[tuple(pos0)] = ""
            self.players[agent] = np.array([*pos])
            delta = np.sum(abs(self.players[agent] -
                               self.players[self.get_enemy(agent)]))
            return delta0 - delta
        return -5

    def attack(self, agent):
        pos = self.players[agent].copy()
        other = self.get_enemy(agent)
        dist = self.players[other].copy() - pos
        if all(abs(dist) <= 1) and not all(dist == 0):
            other.healt -= 1
            if other.healt == 0:
                return 10
            return +2
        return -1

    def do_action(self, agent, fn):
        if agent.healt > 0:
            return fn(), False
        return -10, True

    def get_enemy(self, agent):
        for a in self.players.keys():
            if a.char != agent.char:
                return a
        return None

    def save_qtable(self):
        for p in self.players.keys():
            p.save_qtable()
