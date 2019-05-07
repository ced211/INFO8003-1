import os
import numpy as np

import catcher
import Display

class Continuous_Agent:
	def __init__(self, pretrained_model = None, learning_rate = 0.0001, std = 1, screen_size = None, display = True):
		"""
		Init the agent with the necessary values
		"""
		self.name = 'policy gradient agent gaussian linear'
		if screen_size is None:
			self.screen_size = [400, 400]
		else:
			self.screen_size = screen_size
		self.env = catcher.ContinuousCatcher(width=self.screen_size[0], height=self.screen_size[1])
		if pretrained_model:
			# linear model
			self.theta = pretrained_model
		else:
			# linear model
			w_bar_center, w_bar_vel, w_fruit_center0, w_fruit_center1 = 0, 0, 0, 0
			self.theta = [w_bar_center, w_bar_vel, w_fruit_center0, w_fruit_center1]
		# learning rate for the update
		self.learning_rate  = learning_rate
		# standard deviation of the Gaussian.
		self.std = std
		self.display = display
		if display:
			self.Display = Display.Display()
			
	def step(self, states):
		"""
		return the action to take given the state.
		"""
		list = [state*par for state,par in zip(states,self.theta)]
		mean = np.sum(list)
		action = np.random.normal(mean, self.std)
		return mean, action
		
	def Dlog(self, reward_sum, episode_mean, episode_action, episode_state):
		"""
		return the sum of the derivative of the log policy multiply by the sum of reward.
		"""
		# compute the difference between the action and the mean of the gaussian model divide by the 
		# variance and multiply by the sum of reward
		dif = [(x1 - x2)/(self.std**2)*reward_sum/len(episode_action) for (x1, x2) in zip(episode_action, episode_mean)]
		dlog = [a*b for a,b in zip(episode_state,dif)]
		# sum for all weights
		dlog_array = np.asarray(dlog)
		return np.sum(dlog_array, axis=0)
		
	def update(self, dlog_list, batch_size):
		"""
		Update the parameters of the model.
		"""
		dlog = np.asarray(dlog_list)
		mean = np.mean(dlog, axis=0)
		self.theta = [x1 + self.learning_rate*x2 for (x1, x2) in zip(self.theta, mean)]
		
	def evaluate(self, EPISODE_NUMBER = 50, TIME_LIMITE=1000):
		"""
		Evaluate the model: return the mean of cumulative reward 
		"""
		reward_sum = 0
		episode_number = 0
		previous_observation = self.env.reset()
		sum_reward_list = []
		while episode_number != EPISODE_NUMBER:
			# give the mean and action given an obervation.
			mean, action = self.step(previous_observation)
			# give the observation and reward given an action.
			observation, reward, done = self.env.step([action])
			reward_sum += reward 
			time = 0
			if done or time > TIME_LIMITE:
				sum_reward_list.append(reward_sum)
				reward_sum = 0
				episode_number += 1
				previous_observation = self.env.reset()	
		sum_reward_array = np.asarray(sum_reward_list)
		return np.mean(sum_reward_array), np.std(sum_reward_array)
		
	def train(self, EPISODE_NUMBER = 1000, batch_size = 20, TIME_LIMITE = 1000):
		# Init
		episode_rewards, episode_state, episode_action, episode_mean =[], [], [], []
		episode_number = 1
		reward_sum = 0
		action = [0]
		dlog_list = []
		previous_observation = self.env.reset()
		sum_reward_list = []
		mean_sum_reward_list = []
		# Training progress
		time = 0		
		while episode_number != EPISODE_NUMBER:
			# create dataset
			# give the mean and action given an obervation.
			mean, action[0] = self.step(previous_observation)
			# give the observation and reward given an action.
			observation, reward, done = self.env.step(action)
			reward_sum += (reward * self.env.gamma() ** time)
			episode_state.append(previous_observation)
			episode_action.append(action[0])
			episode_mean.append(mean)
			previous_observation = observation
			if self.display:
				self.Display.draw( self.env.bar.center, self.env.fruit.center, reward_sum, self.env.lives)
			time += 1
			# check if the episode of the game end or does not die after 10000 steps.
			if done or time > TIME_LIMITE:
				nb_element = 0
				dlog = self.Dlog(reward_sum, episode_mean, episode_action, episode_state)
				dlog_list.append(dlog)
				sum_reward_list.append(reward_sum)
				if episode_number % batch_size == 0:
					self.update(dlog_list, batch_size)
					sum_reward_array = np.asarray(sum_reward_list)
					mean_sum_reward = np.mean(sum_reward_array, axis=0)
					mean_sum_reward_list.append(mean_sum_reward)
					dlog_list = []
					sum_reward_list = []
					print(self.theta)
				reward_sum = 0
				time = 0
				episode_number +=1
				episode_rewards, episode_state, episode_action, episode_mean =[], [], [], []
				observation = self.env.reset()
		return mean_sum_reward_list

class Discrete_Agent:
	def __init__(self, pretrained_model = None, learning_rate = 0.0001, std = 1, screen_size = None, display = True):
		"""
		Init the agent with the necessary values
		"""
		self.name = 'policy gradient agent gaussian linear'
		if screen_size is None:
			self.screen_size = [400, 400]
		else:
			self.screen_size = screen_size
		self.env = catcher.ContinuousCatcher(width=self.screen_size[0], height=self.screen_size[1])
		if pretrained_model:
			# linear model
			self.theta = pretrained_model
		else:
			# linear model
			w_bar_center, w_bar_vel, w_fruit_center0, w_fruit_center1 = 0, 0, 0, 0
			self.theta = [w_bar_center, w_bar_vel, w_fruit_center0, w_fruit_center1]
		# learning rate for the update
		self.learning_rate  = learning_rate
		# standard deviation of the Gaussian.
		self.std = std
		self.display = display
		if display:
			self.Display = Display.Display()
			
	def step(self, states):
		"""
		return the action to take given the state.
		"""
		list = [state*par for state,par in zip(states,self.theta)]
		mean = np.sum(list)
		action = np.random.normal(mean, self.std)
		action = int(round(action))
		return mean, action
		
	def Dlog(self, reward_sum, episode_mean, episode_action, episode_state):
		"""
		return the sum of the derivative of the log policy multiply by the sum of reward.
		"""
		# compute the difference between the action and the mean of the gaussian model divide by the 
		# variance and multiply by the sum of reward
		dif = [(x1 - x2)/(self.std**2)*reward_sum for (x1, x2) in zip(episode_action, episode_mean)]
		dlog = [a*b for a,b in zip(episode_state,dif)]
		# sum for all weights
		dlog_array = np.asarray(dlog)
		return np.sum(dlog_array, axis=0)
		
	def update(self, dlog_list, batch_size):
		"""
		Update the parameters of the model.
		"""
		dlog = np.asarray(dlog_list)
		mean = np.mean(dlog, axis=0)
		self.theta = [x1 + self.learning_rate*x2 for (x1, x2) in zip(self.theta, mean)]
		
	def evaluate(self, EPISODE_NUMBER = 50, TIME_LIMITE=1000):
		"""
		Evaluate the model: return the mean of cumulative reward 
		"""
		reward_sum = 0
		episode_number = 0
		previous_observation = self.env.reset()
		sum_reward_list = []
		while episode_number != EPISODE_NUMBER:
			# give the mean and action given an obervation.
			mean, action = self.step(previous_observation)
			# give the observation and reward given an action.
			observation, reward, done = self.env.step([action])
			reward_sum += reward 
			time = 0
			if done or time > TIME_LIMITE:
				sum_reward_list.append(reward_sum)
				reward_sum = 0
				episode_number += 1
				previous_observation = self.env.reset()	
		sum_reward_array = np.asarray(sum_reward_list)
		return np.mean(sum_reward_array), np.std(sum_reward_array)
		
	def train(self, EPISODE_NUMBER = 2000, batch_size = 20, TIME_LIMITE = 1000):
		# Init
		episode_rewards, episode_state, episode_action, episode_mean =[], [], [], []
		episode_number = 1
		reward_sum = 0
		action = [0]
		dlog_list = []
		previous_observation = self.env.reset()
		sum_reward_list = []
		mean_sum_reward_list = []
		# Training progress
		time = 0		
		while episode_number != EPISODE_NUMBER:
			# create dataset
			# give the mean and action given an obervation.
			mean, action[0] = self.step(previous_observation)
			# give the observation and reward given an action.
			observation, reward, done = self.env.step(action)
			reward_sum += (reward * self.env.gamma() ** time)
			episode_state.append(previous_observation)
			episode_action.append(action[0])
			episode_mean.append(mean)
			previous_observation = observation
			if self.display:
				self.Display.draw( self.env.bar.center, self.env.fruit.center, reward_sum, self.env.lives)
			time += 1
			# check if the episode of the game end or does not die after 10000 steps.
			if done or time > TIME_LIMITE:
				nb_element = 0
				dlog = self.Dlog(reward_sum, episode_mean, episode_action, episode_state)
				dlog_list.append(dlog)
				sum_reward_list.append(reward_sum)
				if episode_number % batch_size == 0:
					self.update(dlog_list, batch_size)
					sum_reward_array = np.asarray(sum_reward_list)
					mean_sum_reward = np.mean(sum_reward_array, axis=0)
					mean_sum_reward_list.append(mean_sum_reward)
					dlog_list = []
					sum_reward_list = []
					print(self.theta)
				reward_sum = 0
				time = 0
				episode_number +=1
				episode_rewards, episode_state, episode_action, episode_mean =[], [], [], []
				observation = self.env.reset()
		return mean_sum_reward_list

if __name__ == "__main__":
	agent = Continuous_Agent()
	print(agent.evaluate())
	mean_sum_reward_list = agent.train()
	print(agent.evaluate())
	print(agent.theta)
	