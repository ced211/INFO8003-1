from BaseAgent import BaseAgent
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from RandomAgent import RandomAgent
from catcher import ContinuousCatcher
from emulator import Emulator
from joblib import dump, load
import random

class QAgent(BaseAgent):
    estimator = None
    Fmax =10
    gamma=0.95
    discrete_step=0.1

    def __init__(self,env= ContinuousCatcher(),discrete_step=0.1 ,Fmax = 5,gamma = 0.99,estimator=ExtraTreesRegressor(
        n_estimators=1, max_depth=None, min_samples_split=2, random_state=1997, n_jobs=1)):
        self.history = []        
        self.Fmax = Fmax
        self.estimator = estimator
        self.gamma = gamma
        random_agent = RandomAgent(Fmax,env)
        history = random_agent.play()
        self.history.extend(history)
        self.env=env
        self.discrete_step=discrete_step
        self.name = "Fitted Q agent"
 
    def value(self,s):
        actions = np.arange(- self.Fmax,self.Fmax,self.discrete_step)
        maximum = - float("inf")
        for action in actions:
            prediction = self.estimator.predict(np.append(s,action).reshape(1,-1))
            if prediction > maximum:
                maximum = prediction
        return maximum

    def step(self,s):
        actions = np.arange(- self.Fmax,self.Fmax,self.discrete_step)
        maximum = - float("inf")
        for action in actions:
            prediction = self.estimator.predict(np.append(s,action).reshape(1,-1))
            if prediction > maximum:
                maximum = prediction
                opt_action = []
            if prediction >= maximum:
                opt_action.append(action)

        return random.choice(opt_action)  

    def train(self,n,history=None):
        """train the agent given the history
        The history is a list of tuple: (state,action,reward,next_state)
        state is a numpy array: [bar_center_x, bar_velocity, fruit_center_x, fruit_center_y]
        reward,action are real.
        """
        if history != None:
            self.history.extend(history)
        if len(self.history) == 0:
            print("Error: emtpy batch !")
            return
        exp_replay = min(len(self.history),100000)
        print(exp_replay)
        h = random.choice(self.history)
        X = np.append(h[0],h[1])
        Y = np.asarray(h[2])      
        rewards = np.asarray(h[2])
        index = np.random.choice(len(self.history),(exp_replay,))
        for i in index:
            h = self.history[i]
            X = np.vstack((X,np.append(h[0],h[1])))
            Y = np.vstack((Y,np.asarray(h[2])))     
            rewards = np.vstack((rewards,np.asarray(h[2])))
        Y = np.ravel(Y)
        print("fitting")
        self.estimator.fit(X, Y)
        for j in range(n):
            for i in range(len(Y)-1):
                value = self.value(X[i+1,:4])
                Y[i] = rewards[i] + self.gamma * value
            self.estimator.fit(X,Y)
            print("fitted  " + str(j) + " Q function")

if __name__ == "__main__":
    episode = 0
    agent = QAgent()
    random_agent = RandomAgent()
    while episode < 1000:
        history = random_agent.play()
        agent.history.extend(history)
        episode += 1
        if episode % 10 == 0:
            print("played: " + str(episode) + " episode")
            agent.train(n=20)
            dump(agent, 'FQIagent_' + str(episode) + '.joblib') 
            Emulator.emulate(agent)
            print("number of tree:" + str(len(agent.estimator)))
            np.save("FQI_history_episode_" + str(episode),np.asarray(agent.history))






    
