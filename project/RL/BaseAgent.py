class BaseAgent:
    def __init__(self):
        """
        Init the agent with the necessary values
        """
        self.name = 'BaseAgent'

    def init(self, world_infos=None):
        """
        Initiliaze the agent from world information's

        :param world_info: array-like as [window_size, fruit_size, bar_size]
                            where each entry is a pair of sizes
        """
        pass

    def step(self, state):
        """
        Takes one step based on the current state of the game

        :param state: The state of the game

        :return: The action to take (a single value left/right)
        """
        bar_x = state[0]
        fruit_x = state[2]

        action = fruit_x - bar_x

        return action
