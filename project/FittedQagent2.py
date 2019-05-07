import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import ExtraTreesRegressor
import sys
from sklearn.externals import joblib




def fitted_Q (data_path,action_list, gamma = 0.95,n_iterations = 20):
	"""
	Fitted-Q algorithm.
	"""
	#Get the data
	data = np.load(data_path)

	# [bar_center_x, bar_velocity, fruit_center_x, fruit_center_y, action]
	X = data[:,:5]
	#rewards
	y = data[:,5]
	model = ExtraTreesRegressor(n_estimators = 10, n_jobs=-1)

	#First training: Q0
	model.fit(X,y)
	joblib.dump(model,"Q"+str(1)+".sav")
	for i in range(n_iterations):
		print("Q"+str(i))
		predictions = []
		batch = []
		batch_size = 200

		for j, line in enumerate(data[:,6:]):
			# test all possible actions for the next state.

			# append the next states and all the possible actions to the batch.
			batch.extend([np.append(line,a) for a in action_list])


			if j % batch_size == batch_size-1 or j == len(data)-1:

				# Q value for the states and actions in the batch
				Q_values = model.predict(batch)

				predictions.extend(Q_values)

				# reset pour le prochain batch
				batch = []

		best_prediction = []
		nb_actions = len(action_list)

		
		for k in range(len(y)-1):
			# find the best Q value for a state given all the actions.
			best_prediction.append(max(predictions[k * nb_actions : (k+1) * nb_actions]))
		best_prediction.append(max(predictions[-nb_actions:]))
			
		#Nouveau y = récompense + gamma*prédiction de la récompense max d'après
		y = data[:,5] + gamma* np.array(best_prediction)
		model.fit(X,y)
		joblib.dump(model,"Q"+str(i+2)+".sav")
		
def fitted_Q_pursue_training (data_path,model_path,Q_number,action_list, gamma = 0.95,n_iterations = 1):
	"""
	Fitted-Q algorithm.
	"""
	model = joblib.load(model_path)
	
	
	#Get the data
	data = np.load(data_path)

	# [bar_center_x, bar_velocity, fruit_center_x, fruit_center_y, action]
	X = data[:,:5]
	#rewards
	y = data[:,5]

	for i in range(n_iterations):
		print("Q"+str(i))
		predictions = []
		batch = []
		batch_size = 200

		for j, line in enumerate(data[:,6:]):
			# test all possible action for the next states.

			# append the next states and all the possible actions to the batch.
			batch.extend([np.append(line,a) for a in action_list])


			if j % batch_size == batch_size-1 or j == len(data)-1:

				# Q value for the states and actions in the batch
				batch_predictions = model.predict(batch)

				predictions.extend(batch_predictions)

				# reset pour le prochain batch
				batch = []

		best_prediction = []
		nb_actions = len(action_list)

		
		for k in range(len(y)-1):
			# find the best Q value for a state given all the actions.
			best_prediction.append(max(predictions[k * nb_actions : (k+1) * nb_actions]))
		best_prediction.append(max(predictions[-nb_actions:]))
			
		#Nouveau y = récompense + gamma*prédiction de la récompense max d'après
		y = data[:,5] + gamma* np.array(best_prediction)
		model.fit(X,y)
		joblib.dump(model,"Q"+str(Q_number+i+1)+".sav")

	
fitted_Q("./FQI_history.npy", [-80,-60,-40,-20,-10,-5,0,5,10,20,40,60,80])
#fitted_Q_pursue_training("./FQI_history.npy","Q21.sav",21,[-80,-60,-40,-20,-10,-5,0,5,10,20,40,60,80])