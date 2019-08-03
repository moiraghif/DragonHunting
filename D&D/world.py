import constants
import numpy as np


def die(sides):
    def roll(dies):
        return np.sum([np.random.randint(sides) + 1
                       for _ in range(dies)])
    return roll


d20 = die(20)
d6 = die(6)


class World:
    def __init__(self, players):
        self.world = constants.WORLD.copy()
        self.dim_y, self.dim_x = self.world.shape
        self.players = dict()
        for p in players:
            p.initialize_world(self)
            self.players[p] = self.get_position(p)

    def is_free(self, pos):
        if any(np.array(pos) >= [self.dim_y, self.dim_x]) or \
           any(np.array(pos) < 0):
            return False
        return self.world[pos] == ""

    def get_position(self, agent):
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                if self.world[y, x] == agent.char:
                    return np.array([y, x])
        return False

    def move_of(self, agent, x=0, y=0):
        pos0 = self.players[agent]
        pos = tuple(pos0 + [y, x])
        pos0 = tuple(pos0)
        if self.is_free(pos):
            self.world[pos], self.world[pos0] = agent.char, ""
            self.players[agent] = np.array(list(pos))
            return 0
        return -1

    def attack(self, agent, x=0, y=0):
        pos0 = self.players[agent]
        pos = pos0 + [y, x]
        other = None
        for a in self.players.keys():
            if all(self.players[a] == pos):
                other = a
                party = -1 if agent.party == other.party else 1
                break
        if not other:
            return -1
        if agent.attack + d20(1) > other.defense:
            damage = d6(2)
            other.hp -= damage
            if other.hp <= 0:
                self.die(agent)
                return party * other.max_hp
            return party * damage
        return party

    def do_action(self, agent, fn, *args):
        reward = fn(*args)
        pos = self.players[agent]
        done = all(map(lambda a: a.party == agent.party,
                       self.players.keys()))
        return tuple(pos), reward, done

    def die(self, agent):
        self.world[self.players[agent]] = ""
        del self.players[agent]
