from BaseAgent import BaseAgent
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from perfect_agent import perfectAgent
from RandomAgent import RandomAgent
from catcher import ContinuousCatcher
from emulator import Emulator
from joblib import dump, load
from sklearn.linear_model import SGDRegressor
import random
class QAgent(BaseAgent):
    estimator = None
    gamma=0.95

    def __init__(self,env= ContinuousCatcher(),gamma = 0.99,estimator=ExtraTreesRegressor(
        n_estimators=10,max_depth=None, min_samples_split=2, random_state=1997, n_jobs=-1)):
        self.history = []        
        self.estimator = estimator
        self.gamma = gamma
        self.env=env
        self.name = "Fitted Q agent"
        self.actions = [env.low(),env.high()]

    def step(self,s):
        """return the action to take"""
        input = []
        for action in self.actions:
            input.append(np.append(s,action))
        input = np.asarray(input)
        prediction = self.estimator.predict(input)
        return self.actions[np.argmax(prediction)]

    def train(self,N,history=None):
        """train the agent given the history
        The history is a list of tuple: (state,action,reward,next_state)
        state is a numpy array: [bar_center_x, bar_velocity, fruit_center_x, fruit_center_y]
        reward,action are real.
        """
        nb_sample = len(history)
        history = np.asarray(history)
        #print("history")
        #print(history[:,3])
        X = []
        next_X = []
        reward = history[:,2]          
        for h in history:
            X.append(np.append(h[0],h[1]))
            next_X.append(h[3])
        X = np.asarray(X)
        next_X = np.asarray(next_X)
        #normalize reward
        #reward_std = np.std(reward)
        #reward = np.subtract(reward,np.mean(reward))
        #reward = np.divide(reward,reward_std)
        Q_input = []
        for x in next_X:
            for a in self.actions:
                Q_input.append(np.append(x,a))
        Q_input = np.asarray(Q_input)
        print(Q_input)

        self.estimator.fit(X,reward)
        for n in range(N):    
            predictions = self.estimator.predict(Q_input)
            predictions = predictions.reshape((nb_sample,len(self.actions)))
            print(predictions[0,:])
            max_action_index = np.argmax(predictions,axis=1)
            value = predictions[np.arange(len(predictions)),max_action_index]
            Y = np.add(reward,self.gamma*value)
            self.estimator.fit(X,Y)

if __name__ == "__main__":
    agent = QAgent()
    random_agent = RandomAgent()
    history = random_agent.play(games=1000)
    print("training agent")
    agent.train(50,history)
    dump(agent, 'FQIagent_borned12_22' + '.joblib') 
    Emulator.emulate(agent)






    
