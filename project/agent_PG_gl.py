import os
import numpy as np
from sklearn.preprocessing import StandardScaler
import catcher
import Display

class Continuous_Agent:
	def __init__(self, pretrained_model = None, learning_rate = 0.1, std = 1, screen_size = None, display = True):
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
		self.max_mean_sum_reward = 0
			
	def step(self, states):
		"""
		return the action to take given the state.
		"""
		list = [state*par for state,par in zip(states,self.theta)]
		mean = np.sum(list)
		action = np.random.normal(mean, self.std)
		return mean, action
		
	def Dlog(self, episode_mean, episode_action, episode_state):
		"""
		return the sum of the derivative of the log policy.
		"""
		# compute the difference between the action and the mean of the gaussian model divide by the 
		# variance and multiply by the sum of reward
		dif = [(x1 - x2)/(self.std**2) for (x1, x2) in zip(episode_action, episode_mean)]
		dlog = [a*b for a,b in zip(episode_state,dif)]
		# sum for all weights
		dlog_array = np.asarray(dlog)
		return np.mean(dlog_array, axis=0)
		
	def update(self, dlog_list,sum_reward_list):
		"""
		Update the parameters of the model.
		"""
		dlog = np.asarray(dlog_list)
		scaler = StandardScaler()
		norm_reward = scaler.fit_transform(np.asarray(sum_reward_list).reshape(-1,1))
		dlog = np.multiply(dlog,norm_reward)
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
		time = 0
		while episode_number != EPISODE_NUMBER:
			# give the mean and action given an obervation.
			mean, action = self.step(previous_observation)
			# give the observation and reward given an action.
			observation, reward, done = self.env.step([action])
			reward_sum += reward
			time += 1
			if done or time > TIME_LIMITE:
				sum_reward_list.append(reward_sum)
				reward_sum = 0
				episode_number += 1
				previous_observation = self.env.reset()	
				time = 0
			previous_observation = observation
		sum_reward_array = np.asarray(sum_reward_list)
		return np.mean(sum_reward_array), np.std(sum_reward_array)
		
	def train(self, EPISODE_NUMBER = 200000, batch_size = 1000, TIME_LIMITE = 1000):
		"""
		Train the model.
		"""
		# Init
		print("continous agent")
		episode_rewards, episode_state, episode_action, episode_mean =[], [], [], []
		episode_number = 1
		reward_sum= 0
		rewards= []
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
			reward_sum += reward
			rewards.append(reward)
			episode_state.append(previous_observation)
			episode_action.append(action[0])
			episode_mean.append(mean)
			previous_observation = observation
			if self.display:
				self.Display.draw( self.env.bar.center, self.env.fruit.center, reward_sum, self.env.lives)
			time += 1
			# check if the episode of the game end or does not die after TIME_LIMITE steps.
			if done or time > TIME_LIMITE:
				dlog = self.Dlog(episode_mean, episode_action, episode_state)
				dlog_list.append(dlog)
				sum_reward_list.append(reward_sum)
				rewards= []
				if episode_number % batch_size == 0:
					self.update(dlog_list,sum_reward_list)
					sum_reward_array = np.asarray(sum_reward_list)
					mean_sum_reward = np.mean(sum_reward_array, axis=0)
					if mean_sum_reward > self.max_mean_sum_reward:
						diff = mean_sum_reward - self.max_mean_sum_reward
						coeff = (100 + diff) / 100
						if self.max_mean_sum_reward > 0:
							self.learning_rate /= coeff
						self.max_mean_sum_reward = mean_sum_reward

						print("learning rate: " + str(self.learning_rate))
					print(mean_sum_reward)
					print(self.max_mean_sum_reward)					
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
	def __init__(self, pretrained_model = None, std=1,learning_rate = 0.1, screen_size = None, display = True):
		"""
		Init the agent with the necessary values
		"""
		self.name = 'discrete policy gradient agent gaussian linear'
		self.max_mean_sum_reward = 0
		if screen_size is None:
			self.screen_size = [400, 400]
		else:
			self.screen_size = screen_size
		self.env = catcher.ContinuousCatcher(width=self.screen_size[0], height=self.screen_size[1])
		print(self.env.high())
		self.std=self.env.high()/10
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
		action = round(action/0.1)*0.1
		return mean, action
		
	def Dlog(self, reward_sum, episode_mean, episode_action, episode_state):
		"""
		return the sum of the derivative of the log policy.
		"""
		# compute the difference between the action and the mean of the gaussian model divide by the 
		# variance and multiply by the sum of reward
		dif = [(x1 - x2)/(self.std**2) for (x1, x2) in zip(episode_action, episode_mean)]
		dlog = [a*b for a,b in zip(episode_state,dif)]
		# sum for all weights
		dlog_array = np.asarray(dlog)
		return np.mean(dlog_array, axis=0)
		
	def update(self, dlog_list, sum_reward_list):
		"""
		Update the parameters of the model.
		"""
		dlog = np.asarray(dlog_list)
		scaler = StandardScaler()
		norm_reward = scaler.fit_transform(np.asarray(sum_reward_list).reshape(-1,1))
		dlog = np.multiply(dlog,norm_reward)
		mean = np.mean(dlog, axis=0)
		self.theta = [x1 + self.learning_rate*x2 for (x1, x2) in zip(self.theta, mean)]
		
	def evaluate(self, EPISODE_NUMBER = 50, TIME_LIMITE=10000):
		"""
		Evaluate the model: return the mean of cumulative reward 
		"""
		reward_sum = 0
		episode_number = 0
		previous_observation = self.env.reset()
		sum_reward_list = []
		time = 0
		while episode_number != EPISODE_NUMBER:
			# give the mean and action given an obervation.
			mean, action = self.step(previous_observation)
			# give the observation and reward given an action.
			observation, reward, done = self.env.step([action])
			reward_sum += reward
			time += 1
			if done or time > TIME_LIMITE:
				sum_reward_list.append(reward_sum)
				reward_sum = 0
				episode_number += 1
				previous_observation = self.env.reset()
				time = 0
			previous_observation = observation
		sum_reward_array = np.asarray(sum_reward_list)
		return np.mean(sum_reward_array), np.std(sum_reward_array)
		
	def train(self, EPISODE_NUMBER = 20000, batch_size = 1, TIME_LIMITE = 1000):
		"""
		Train the model.
		"""
		# Init
		print("discrete gent")
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
			reward_sum += reward 
			episode_state.append(previous_observation)
			episode_action.append(action[0])
			episode_mean.append(mean)
			previous_observation = observation
			if self.display:
				self.Display.draw( self.env.bar.center, self.env.fruit.center, reward_sum, self.env.lives)
			time += 1
			# check if the episode of the game end or does not die after TIME_LIMITE steps.
			if done or time > TIME_LIMITE:
				nb_element = 0
				dlog = self.Dlog(reward_sum, episode_mean, episode_action, episode_state)
				dlog_list.append(dlog)
				sum_reward_list.append(reward_sum)
				if episode_number % batch_size == 0:
					self.update(dlog_list, sum_reward_list)
					sum_reward_array = np.asarray(sum_reward_list)
					mean_sum_reward = np.mean(sum_reward_array, axis=0)
					mean_sum_reward_list.append(mean_sum_reward)
					if mean_sum_reward > self.max_mean_sum_reward:
						diff = mean_sum_reward - self.max_mean_sum_reward
						coeff = (100 + diff) / 100
						if self.max_mean_sum_reward > 0:
							self.learning_rate /= coeff
						self.max_mean_sum_reward = mean_sum_reward
						print("learning rate: " + str(self.learning_rate))
					print(mean_sum_reward)
					print(self.max_mean_sum_reward)	
					dlog_list = []
					sum_reward_list = []				
				reward_sum = 0
				time = 0
				episode_number +=1
				episode_rewards, episode_state, episode_action, episode_mean =[], [], [], []
				observation = self.env.reset()
				
		return mean_sum_reward_list

if __name__ == "__main__":
	import matplotlib.pyplot as plt
	import numpy as np
	
	continuous_agent = Continuous_Agent(display = False)
	discrete_agent = Discrete_Agent(display = False)
	continuous_mean,continuous_std = continuous_agent.evaluate()
	continuous_mean_list = []
	continuous_std_list = []
	continuous_mean_list.append(continuous_mean)
	continuous_std_list.append(continuous_std)
	continuous_mean_sum_reward_list = []
	discrete_mean,discrete_std = discrete_agent.evaluate()
	discrete_mean_list = []
	discrete_std_list = []
	discrete_mean_list.append(discrete_mean)
	discrete_std_list.append(discrete_std)
	discrete_mean_sum_reward_list = []
	
	for i in range(5):
		continuous_agent = Continuous_Agent(display = False)
		discrete_agent = Discrete_Agent(display = False)
		print(i)
		continuous_mean_sum_reward_list.extend(continuous_agent.train())
		continuous_mean,continuous_std = continuous_agent.evaluate()
		continuous_mean_list.append(continuous_mean)
		continuous_std_list.append(continuous_std)
		discrete_mean_sum_reward_list.extend(discrete_agent.train())
		discrete_mean,discrete_std = discrete_agent.evaluate()
		discrete_mean_list.append(discrete_mean)
		discrete_std_list.append(discrete_std)
		np.save("continuous_std_list",np.asarray(continuous_std_list))
		np.save("continuous_mean_list",np.asarray(continuous_mean_list))
		np.save("continuous_mean_sum_reward_list",np.asarray(continuous_mean_sum_reward_list))
		np.save("discrete_std_list",np.asarray(discrete_std_list))
		np.save("discrete_mean_list",np.asarray(discrete_mean_list))
		np.save("discrete_mean_sum_reward_list",np.asarray(discrete_mean_sum_reward_list))
	
	
	# Plot the data sum reward
	continuous_mean_sum_reward_list_list = np.split(np.load("continuous_mean_sum_reward_list.npy"),4)
	discrete_mean_sum_reward_list_list = np.split(np.load("discrete_mean_sum_reward_list.npy"),4)
	for i in range(4):
		plt.plot(continuous_mean_sum_reward_list_list[i], label='continuous agent')
		plt.xlabel("episode")
		plt.title("Average sum reward")
		plt.plot(discrete_mean_sum_reward_list_list[i], label='discrete agent')
		plt.legend()
		plt.show()
	

	continuous_mean_list = np.load("continuous_mean_list.npy")
	continuous_std_list = np.load("continuous_std_list.npy")
	print(continuous_mean_list.shape)
	plt.errorbar(np.arange(5),continuous_mean_list,continuous_std_list,fmt="o")
	plt.xlabel("model")
	plt.ylabel("Average sum reward")
	plt.title("Continous agent")
	plt.show()

	discrete_mean_list = np.load("discrete_mean_list.npy")
	discrete_std_list = np.load("discrete_std_list.npy")
	plt.errorbar(np.arange(5),discrete_mean_list,discrete_std_list,fmt="o")
	plt.xlabel("model")
	plt.ylabel("Average sum reward")
	plt.title("Discrete agent")
	plt.show()




	