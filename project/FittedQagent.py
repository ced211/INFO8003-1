from BaseAgent import BaseAgent
from sklearn.ensemble import ExtraTreesRegressor
import numpy as np

class QAgent(BaseAgent):
    estimator = None
    Fmax =10
    discret_step =50
    gamma=0.95

    def __init__(self, Fmax = 100,step= 50,gamma = 0.95,estimator=ExtraTreesRegressor(
        n_estimators=1, max_depth=None, min_samples_split=2, random_state=1997, n_jobs=-1)):
        self.Fmax = Fmax
        self.discret_step=step
        self.estimator = estimator
        self.gamma = gamma

    def step(self,s):
        actions = np.arange(- self.Fmax,self.Fmax,self.discret_step)
        maximum = - float("inf")
        for action in actions:
            prediction = self.estimator.predict(np.asarray((s,action)).reshape(1, -1))
            if prediction > maximum:
                maximum = prediction
                opt_action = action
        return opt_action,maximum

    def fit(self,history,n):
        """return the estimation of Q given the history
        The history is a list of tuple: (state,action,reward,next_state)
        state is a numpy array: [bar_center_x, bar_velocity, fruit_center_x, fruit_center_y]
        reward,action are real.
        """
        X = []
        Y = []
        for h in history:
            X.append(np.asarray((h[0],h[1])).reshape(1, -1))
            Y.append(h[2])
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.estimator.fit(X, Y)
        for _ in range(n):
            for i in range(len(Y)):
                _,value = self.step(history[i][3])
                Y[i] = h[i] + self.gamma * value
            self.estimator.fit(X,Y)



    
