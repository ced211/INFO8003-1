import math
class BaseAgent:

    def __init__(self,env,name="BaseAgent"):
        self.env=env
        self.name = name
    
    def step(self, state):
        """
        Takes one step based on the current state of the game

        :param state: The state of the game

        :return: The action to take (a single value left/right)
        """
        raise NotImplementedError("subclass should implement this")

    def play(self,max_step=math.inf):
        """Play until terminal state or until taking max_step action, return the history

        The history is a list of tuple: (state,action,reward,next_state)
        """

        print("Agent " + self.name + " is playing.")
        self.env.reset()
        done = False
        history = []
        while not done and max_step > 0:
            cur_state = self.env.observe()
            action = self.step(cur_state)
            next_state,reward,done = self.env.step(action)
            history.append((cur_state,action,reward,next_state))
            max_step -= 1
        print("Agent " + self.name + " has played.")    
        return history
