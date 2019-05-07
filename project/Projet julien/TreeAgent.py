from sklearn.ensemble import ExtraTreesRegressor
import sys
from sklearn.externals import joblib
import numpy as np

class TreeAgent():
    def __init__(self,tree_path):
        self.tree = joblib.load(tree_path)
        self.name = "Fitted-Q"
    def init(self, actions, world_infos=None):
        """
        Initialize the agent from world information's

        :param actions: Actions available
        :param world_infos: array-like as [window_size, fruit_size, bar_size]
                            where each entry is a pair of sizes
        """
        self.actions = actions
        pass
    
    def step(self,state):
        state = np.array(state)
        batch = []
        batch.extend([np.append(state,a) for a in self.actions])
        predictions = self.tree.predict(batch)
        return self.actions[np.argmax(predictions)]