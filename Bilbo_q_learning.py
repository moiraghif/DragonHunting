from CreateBilboWorld import *
import numpy as np
from importlib import reload
import ipdb

TOT_EPISODES=500
MAX_EPOCH = 1000

#initalize the q_table:
possible_moves = {'up':0,'down':1,'left':2,'right':3}
inverse_possible_moves = {0:'up',1:'down',2:'left',3:'right'}
q_table={}
for y in range(WORLD_DIM):
	for x in range(WORLD_DIM):
		for exist_reward in [0,1]:
				q_table[(y,x),exist_reward]=[0,0,0,0]


alpha = 0.5
gamma = 0.3
epsilon = 0.2
decay_epsilon = 0.99
rewards = []

for ep in range(TOT_EPISODES):
	#recreate the environment
	bilbo=Agent(PLAYER_CHAR)
	mondo=World(WORLD_DIM,WORLD_DIM,bilbo=bilbo,obstacle=True)
	#print(World)
	#do Q-stuff
	#print(bilbo.get_pos())
	game_ended=False
	epoch = 0
	try:
		while not game_ended and epoch < MAX_EPOCH:
			#remember to implement the fear later
			current_state,treasure_gone,game_ended = mondo.get_state()
			if np.random.uniform(0,1) < epsilon:
				action = possible_moves[bilbo.random_action()]
				#print(action)
			else:
				action = np.argmax(q_table[current_state,treasure_gone])
				#print(action)

			old_q_val = q_table[current_state,treasure_gone][action]
			bilbo.move(inverse_possible_moves[action])()
			new_state,treasure_gone,game_ended = mondo.get_state()
			reward = mondo.reward()
			if reward == -50:
				new_q_val = reward
				 #ipdb.set_trace()
			elif new_state==current_state:
				new_q_val = -10 #-10 in case of obstacle
			else:
				next_q_val = np.max(q_table[new_state,treasure_gone])
				new_q_val = (1-alpha) * old_q_val + alpha*(reward + gamma*next_q_val)
			q_table[current_state,treasure_gone][action] = new_q_val

			#ipdb.set_trace()


			#bilbo.move(bilbo.random_action())()
			#print(q_table[World.get_state()[0:2]])

			#game_ended = current_state[2]
			#print(current_state)
			#print(game_ended)
			epoch += 1
			#if epoch % 50 == 0:
			#	print("epoch", epoch)

		if ep % 100 == 0:
			print("episode", ep)
		epsilon *= decay_epsilon
		print(mondo)
		print(mondo.get_state())
		print(mondo.reward())
	except:
		#had some issues with some index can be removed later
		#kept it in case there were some errors it's easier to debug
		ipdb.set_trace()

print(q_table)
#print(mondo)
