from catcher import ContinuousCatcher
import random
import math
import numpy as np

class RandomAgent():

	def __init__(self,Fmax = 100.0,env=ContinuousCatcher(width=400, height=400)):
		self.Fmax=Fmax
		self.env=env
		self.name="Random Agent"
    
	def step(self,state):
		"""
		return a random action between [-Fmax and Fmax].
		"""
		
		return (random.random()-0.5) * self.Fmax * 2
	def play(self, games = 4000):
		"""
		return the history of multiple games.
		"""
		state = self.env.reset()
		i = 0
		history = []
		while i != games:
			action = self.step(state)
			next_state, reward, done = self.env.step([action])
			history.append([state[0], state[1], state[2], state[3], action, reward, next_state[0], next_state[1], next_state[2], next_state[3]])
			state = next_state
			if done:
				i += 1
				state = self.env.reset()
		return history

if __name__ == "__main__":
	history = RandomAgent().play()
	print(len(history))
	np.save("FQI_history",np.asarray(history))
	