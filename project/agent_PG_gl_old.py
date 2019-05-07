import os
import numpy as np

import catcher
import Display

class Agent:
	def __init__(self, pretrained_model = None, model_path ="/models", screen_size = None, display = True):
		"""
		Init the agent with the necessary values
		"""
		self.name = 'policy gradient agent gaussian lineaire'
		self.model_path = model_path
		if screen_size is None:
			self.screen_size = [400, 400]
		else:
			self.screen_size = screen_size
		self.env = catcher.ContinuousCatcher(width=self.screen_size[0], height=self.screen_size[1])
		if pretrained_model:
			self.model = load_model(model_path)
		else:
			# give a learning rate 
			self.learning_rate  = 0.001
			# linear model
			w_bar_center, w_bar_vel, w_fruit_center0, w_fruit_center1 = -1, 0, 1, 0
			self.theta = [w_bar_center, w_bar_vel, w_fruit_center0, w_fruit_center1]
			self.std = 1
		self.display = display
		if display:
			self.Display = Display.Display()
			
	def step(self, state):
		"""
		return the action to take given the state.
		"""
		list = [a*b for a,b in zip(state,self.theta)]
		mean = np.sum(list)
		action = np.random.normal(mean, self.std)
		return mean, action
		
	def train(self):
		# Init
		episode_rewards, episode_state, episode_action, episode_mean =[], [], [], []
		episode_number = 0
		reward_sum = 0
		action = [0]
		previous_observation = self.env.reset()        
		# Training progress
		time = 0		
		while episode_number != 1000:

			# create dataset
			# give the mean and action given an obervation.
			mean, action = self.step(previous_observation)
			# give the observation and reward given an action.
			# print(action)
			observation, reward, done = self.env.step(action)
			reward_sum += (reward * self.env.gamma() ** time)
			episode_state.append(previous_observation)
			episode_action.append(action)
			episode_mean.append(mean)
			previous_observation = observation
			if self.display:
				
				self.Display.draw( self.env.bar.center, self.env.fruit.center, reward_sum, self.env.lives)
			# check if the episode of the game end.
			time += 1
			if done or time > 10000:
				time = 0
				# compute the sum of rewards
				episode_number +=1
				# compute the difference between the mean and action
				dif = [x1 - x2 for (x1, x2) in zip(episode_mean, episode_action)]
				print("reward_sum",reward_sum)
				# multiply the difference with the state value (derivative of the model). array of array of 4 elements.
				mul = [a*b for a,b in zip(episode_state,dif)]
				nb_element = 0
				add_mul = []
				for l in mul:
					nb_element += 1
					if len(add_mul) == 0:
						add_mul = l
					else:
						add_mul = [x1 + x2 for (x1, x2) in zip(l, add_mul)]
				print("add_mul/nb_element",[a/nb_element for a in add_mul])
				print("nb_element", nb_element)
				dlog = [-1/2*(x1/self.std/nb_element)*reward_sum for x1 in add_mul]
				print("dlog", dlog)
				self.theta = [x1 + self.learning_rate*x2 for (x1, x2) in zip(self.theta, dlog)]
				print("self.theta", self.theta)
				reward_sum = 0
				episode_rewards, episode_state, episode_action, episode_mean =[], [], [], []
				observation = self.env.reset()

if __name__ == "__main__":
    Agent().train()