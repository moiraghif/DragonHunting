from CreateBilboWorld import *
import numpy as np

TOT_EPISODES=100

#initalize the q_table:
q_table={}
for y in range(WORLD_DIM):
	for x in range(WORLD_DIM):
		for exist_reward in [0,1]:
			q_table[y,x,exist_reward]=0


rewards = []

World=World(WORLD_DIM,WORLD_DIM,obstacle=True)
print(World)

for ep in TOT_EPISODES:
	#do Q-stuff