from catcher import ContinuousCatcher
import numpy as np
from FittedQagent import QAgent
from RandomAgent import RandomAgent
from emulator import Emulator

def evaluate(agent,nb_ite):
    
train = np.asarray([10,100,1000])
score = np.zeros(train.shape)
eval_loop = 100
Fmax=1
game = ContinuousCatcher()
random_agent = RandomAgent(1)
agent = QAgent(10,Fmax,game.gamma())
init_emul = Emulator(agent=random_agent)
init_history = init_emul.run()
agent.train(init_history)

i = 0
for nb_train in range(train[-1]):
    if nb_train == train[i]:
        evaluate(agent)


