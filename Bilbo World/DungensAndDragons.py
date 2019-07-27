import numpy as np


WALL_CHAR = "W"
DRAGON_CHAR = "D"
OBSTACLE_CHAR = "X"
TREASURE_CHAR = "T"
PLAYER_CHAR = "M"
PLAYER1_CHAR = "P"
PLAYER2_CHAR = "Q"
ENEMY1_CHAR = "E"
ENEMY2_CHAR = "F"

WORLD = np.array([["" for x in range(20)]
                  for y in range(20)])
WORLD[0:3, 7] = WALL_CHAR
WORLD[2:5, 11] = WALL_CHAR
WORLD[3:6, 15] = WALL_CHAR
WORLD[5:13, 7] = WALL_CHAR
WORLD[5:7, 17] = WALL_CHAR
WORLD[7:10, 13] = WALL_CHAR
WORLD[12:16, 13] = WALL_CHAR
WORLD[15:20, 5] = WALL_CHAR
WORLD[18:20, 13] = WALL_CHAR
WORLD[7, 7:10] = WALL_CHAR
WORLD[7, 12:14] = WALL_CHAR
WORLD[8, 18:20] = WALL_CHAR
WORLD[9, 3:8] = WALL_CHAR
WORLD[12, 7:11] = WALL_CHAR
WORLD[14, 13:16] = WALL_CHAR
WORLD[14, 18:20] = WALL_CHAR
WORLD[0, 15] = WALL_CHAR
WORLD[1, 18] = WALL_CHAR
WORLD[2, 1] = WALL_CHAR
WORLD[3, 9] = WALL_CHAR
WORLD[4, 5] = WALL_CHAR
WORLD[6, 2] = WALL_CHAR
WORLD[9, [0, 9, 16]] = WALL_CHAR
WORLD[12, [3, 16]] = WALL_CHAR
WORLD[15, [8, 10]] = WALL_CHAR
WORLD[16, [2, 15]] = WALL_CHAR
WORLD[17, [8, 10]] = WALL_CHAR
WORLD[18, 18] = WALL_CHAR

WORLD[0, 0], WORLD[19, 19] = PLAYER2_CHAR, PLAYER1_CHAR
WORLD[19, 0], WORLD[0, 19] = ENEMY1_CHAR, ENEMY2_CHAR


def create_die(sides):
    def roll_die(dies):
        return np.sum([np.random.randint(sides) + 1
                       for _ in range(dies)])
    return roll_die


DIES = {n: create_die(n) for n in [4, 6, 8, 12, 20]}
print(DIES[4](2))
