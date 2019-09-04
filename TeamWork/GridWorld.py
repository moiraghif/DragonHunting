import numpy as np
import re
import constants


import constants
import networkx as nx
import numpy as np
import re

OBSTACLE_CHAR = "█"


class World:
    def __init__(self, players, new_qtable):
        self.world = constants.WORLD.copy()
        self.dim_y, self.dim_x = self.world.shape
        self.players = dict()
        self.graph = nx.Graph()
        self.create_graph()
        for p in players:
            self.casual_spawn(p)
            p.initialize_world(self, new_qtable)
        #Here we will put the tombs of the dead
        #it's an isolated place so there should be no trouble
        self.graveyard = [(0,0),(0,1)]

    def create_graph(self):
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                if self.is_free((y, x)):
                    self.graph.add_node(self.pos_to_node((y, x)))
                    if self.pos_to_node((y - 1, x)) in self.graph.nodes:
                        self.graph.add_edge(self.pos_to_node((y - 1, x)),
                                            self.pos_to_node((y, x)))
                    if self.pos_to_node((y, x - 1)) in self.graph.nodes:
                        self.graph.add_edge(self.pos_to_node((y, x - 1)),
                                            self.pos_to_node((y, x)))

    def pos_to_node(self, pos):
        return " ".join(map(str, list(pos)))

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
        #if in grave
        if pos[0] < 3 and pos[1] < 3:
            return False
        return len(self.world[pos]) == 0

    def move_of(self, agent, x=0, y=0):
        delta = np.array([y, x])
        pos0 = self.players[agent]
        pos = tuple(pos0 + delta)
        enemy_dist = self.get_dist_closer_enemy(agent)
        if self.is_free(pos):
            self.world[pos] = agent.char
            self.world[tuple(pos0)] = ""
            self.players[agent] = np.array([*pos])
            delta = enemy_dist - self.get_dist_closer_enemy(agent)
            #return 0
            return 2 if delta > 0 else -2
        return -5

    def attack(self, agent):
        reward = 0
        for enemy in self.players.keys():
            if agent.char == enemy.char or not enemy.alive():
                continue
            try:
                dist = len(nx.shortest_path(self.graph,
                                            self.pos_to_node(
                                                self.players[agent]),
                                            self.pos_to_node(
                                                self.players[enemy])))
            except nx.exception.NodeNotFound:
                return -2
            if dist == 2:
                enemy.healt -= 1
                if enemy.type == agent.type:
                    reward -= 10
                else:
                    reward += (5 if enemy.alive() else 25)
        if reward > 0:
            reward = reward + (agent.healt - 3)/(agent.healt + 1)*reward
        return reward if reward != 0 else -5

    def get_dist_to_enemy(self, agent, enemy):
        return nx.shortest_path_length(self.graph,
                                       self.pos_to_node(self.players[agent]),
                                       self.pos_to_node(self.players[enemy]))

    def get_dist_closer_enemy(self, agent):
        enemies = [enemy for enemy in self.players.keys()
                 if enemy.char != agent.char and enemy.alive() and enemy.type != agent.type]
        if len(enemies) < 1:
            return 0
        return min([
            nx.shortest_path_length(self.graph,
                                    self.pos_to_node(self.players[agent]),
                                    self.pos_to_node(self.players[enemy]))
            for enemy in enemies])

    def get_dist_to_enemies(self, agent):
        enemies = [p for p in self.players.keys()
                   if p.char != agent.char and p.alive() and p.type != agent.type]
        paths = np.array([nx.shortest_path_length(self.graph,
                                                  self.pos_to_node(
                                                      self.players[agent]),
                                                  self.pos_to_node(
                                                      self.players[enemy]))
                          for enemy in enemies])
        return np.min(paths) if len(paths) > 0 else 0

    def get_closer_enemy(self, agent):
        # try:
        pos0 = self.pos_to_node(self.players[agent])
        enemy_pos = [self.pos_to_node(k) for p, k in self.players.items()
                     if p.char != agent.char and p.alive() and p.type != agent.type]
        if not enemy_pos:
            return 0, 0
        pos = [nx.shortest_path(self.graph, pos0, e)
               for e in enemy_pos]
        pos_len = np.array(list(map(len, pos)))
        min_pos = [np.array(list(map(int, p.split(" "))))
                   for p in pos[np.argmin(pos_len)]]
        if not min_pos:
            return 0, 0
        # except nx.exception.NodeNotFound:
        #     return 0, 0
        direction = min_pos[1] - min_pos[0]
        if direction[0] == +1:
            d = 0
        elif direction[0] == -1:
            d = 1
        elif direction[1] == +1:
            d = 2
        elif direction[1] == -1:
            d = 3
        return d, (0 if len(min_pos) == 2 else 1)

    def do_action(self, agent, fn):
        if agent.healt > 0:
            return fn(), False
        if self.pos_to_node(self.players[agent]) in self.graph.nodes:
            self.put_p_in_grave(agent)
            return -10, True
        return 0, True

    def get_pos(self,player):
        char = player.char
        for y in range(self.dim_y):
            for x in range(self.dim_x):
                if char == self.world[y, x]:#agent.char:
                    return (y, x)
        return False

    def put_p_in_grave(self,player):
        for pos in self.graveyard:
            if self.world[pos] == '':
                grave = pos
                break
        actual_pos = self.get_pos(player)
        self.world[grave] = player.char
        self.world[actual_pos] = ''
        self.players[player]=np.array(grave)

    def get_current_state(self, agent):
        others = [p for p in self.players.keys() if p.char != agent.char]

        if not agent.alive():
            state = [0, 0, 0, 0]
            return tuple(state)

        state = []
        health = 1 if agent.healt > 3 else 0
        state.append(health)
        for p in others:
            if not p.alive():
                state.append(0)
                state.append(0)
                continue

            pos0 = self.pos_to_node(self.players[agent])
            enemy_pos = [self.pos_to_node(self.players[p])]
            pos = [nx.shortest_path(self.graph, pos0, e)
                   for e in enemy_pos]
            pos_len = np.array(list(map(len, pos)))
            min_pos = [np.array(list(map(int, p.split(" "))))
                       for p in pos[np.argmin(pos_len)]]

            direction = min_pos[1] - min_pos[0]
            if direction[0] == +1:
                d = 0
            elif direction[0] == -1:
                d = 1
            elif direction[1] == +1:
                d = 2
            elif direction[1] == -1:
                d = 3
            if len(min_pos) == 2:
                dist = 0
            elif len(min_pos) == 3:
                dist = 1
            elif len(min_pos) == 4:
                dist = 2
            else:
                dist = 3
            state.append(d)
            state.append(dist)

        return tuple(state)

    def save_qtable(self):
        for p in self.players.keys():
            p.save_qtable()

    #Deep Learing stuff
    def deep_learning_state(self, agent):
        others = [p for p in self.players.keys() if p.char != agent.char]

        if not agent.alive():
            state = [0, 0, 0, 0, 0]
            return tuple(state)

        state = []
        for p in others:
            if not p.alive():
                state.append(0)
                state.append(0)
                continue

            pos0 = self.pos_to_node(self.players[agent])
            enemy_pos = [self.pos_to_node(self.players[p])]
            pos = [nx.shortest_path(self.graph, pos0, e)
                   for e in enemy_pos]
            pos_len = np.array(list(map(len, pos)))
            min_pos = [np.array(list(map(int, p.split(" "))))
                       for p in pos[np.argmin(pos_len)]]

            direction = min_pos[1] - min_pos[0]
            if direction[0] == +1:
                d = 0
            elif direction[0] == -1:
                d = 1
            elif direction[1] == +1:
                d = 2
            elif direction[1] == -1:
                d = 3
            dist = len(min_pos)
            state.append(d)
            state.append(dist - 1)
        state.append(agent.healt)
        return np.array(state)

    def save_NN(self):
        for p in self.players.keys():
            p.save_NN()

    def __str__(self):
        "Convert the world into string"
        txt = "┌" + "─" * (2*self.dim_x) + "┐\n│"
        txt += '|\n|'.join([''.join(['{:2}'.format(item) for item in row])
                        for row in self.world])
        txt += "│\n" + "└" + "─" * (2*self.dim_x) + "┘"
        txt = re.sub('W\s',2*OBSTACLE_CHAR,txt)
        return txt
