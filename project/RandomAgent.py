from BaseAgent import BaseAgent
from catcher import ContinuousCatcher
import random
import math

class RandomAgent(BaseAgent):

    Fmax = 5.0
    def __init__(self,Fmax=5.0,env=ContinuousCatcher()):
        self.Fmax=Fmax
        self.env=env
        self.name="Random Agent"
    
    def step(self,state):
        return random.random() * self.Fmax

