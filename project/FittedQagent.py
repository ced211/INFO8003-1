from BaseAgent import BaseAgent
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np
from RandomAgent import RandomAgent
from catcher import ContinuousCatcher

class QAgent(BaseAgent):
    estimator = None
    Fmax =10
    gamma=0.95
    discrete_step=100

    def __init__(self,env= ContinuousCatcher(),discrete_step=100 ,Fmax = 5,gamma = 0.99,estimator=ExtraTreesRegressor(
        n_estimators=1, max_depth=None, min_samples_split=2, random_state=1997, n_jobs=-1)):
        self.history = []        
        self.Fmax = Fmax
        self.estimator = estimator
        self.gamma = gamma
        random_agent = RandomAgent(Fmax,env)
        self.train(random_agent.play(),1)
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
                opt_action = action
        return opt_action    

    def train(self,history,n):
        """train the agent given the history
        The history is a list of tuple: (state,action,reward,next_state)
        state is a numpy array: [bar_center_x, bar_velocity, fruit_center_x, fruit_center_y]
        reward,action are real.
        """
        self.history.extend(history)
        X = []
        Y = []
        for h in history:
            X.append(np.append(h[0],h[1]))
            Y.append(h[2])       
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.estimator.fit(X, Y)
        for _ in range(n):
            for i in range(len(Y)):
                value = self.value(history[i][3])
                Y[i] = history[i][1] + self.gamma * value
            self.estimator.fit(X,Y)

if __name__ == "__main__":
    episode = 0
    agent = QAgent()
    while episode < 10000:
        history = agent.play()
        print("score: " + str(history[-1][2]) + " at episode " + str(episode))
        agent.train(history,40)
        episode += 1





    
