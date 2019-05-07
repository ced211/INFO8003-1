import math

import matplotlib.pyplot as plt
import numpy as np
import pygame

import catcher
from BaseAgent import BaseAgent, DiscretizedBaseAgent
from DiscretizedAgent import DiscretizedAgent
from TreeAgent import TreeAgent


class DefaultColorStyle:
    """
    Define the color style of the game
    """

    def __init__(self):
        self.bar_color = (0, 0, 255)
        self.fruit_color = (0, 255, 0)
        self.background_color = (0, 0, 0)
        self.text_color = [255 - self.background_color[i] for i in
                           range(len(self.background_color))]

    def set_bar_color(self, color):
        self.bar_color = color

    def set_fruit_color(self, color):
        self.fruit_color = color

    def set_background_color(self, color):
        self.background_color = color
        self.text_color = [255 - self.background_color[i] for i in
                           range(len(self.background_color))]


class Emulator:
    def __init__(self, options, agent=None):
        """
        Create the emulator with all the necessary variables

        :param options: The options used to set up the emulator
        :param agent: The agent to use (None for user controlled)
        """
        option_tmp = options.get_option("screen_size")

        if option_tmp is None:
            self.screen_size = [400, 400]

        option_tmp = options.get_option("color_style")
        self.color_style = DefaultColorStyle() if option_tmp is None else option_tmp

        option_tmp = options.get_option("max_generations")
        self.max_generations = 1000 if option_tmp is None else option_tmp
        option_tmp = options.get_option("display_generations_gap")
        self.display_generations_gap = 250 if option_tmp is None else option_tmp
        option_tmp = options.get_option("infinite_generations")
        self.infinite_generations = False if option_tmp is None else option_tmp

        option_tmp = options.get_option("fps")
        self.fps = 30 if option_tmp is None else option_tmp

        option_tmp = options.get_option("max_frames")
        self.max_frames = 0 if option_tmp is None else option_tmp

        self.running = True
        self.current_gen = 0
        self.user_right = 0
        self.user_left = 0

        option_tmp = options.get_option("show_display")
        self.display = True if option_tmp is None else option_tmp

        if self.display is True:
            pygame.display.init()
            pygame.font.init()
            self.surface = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Catcher")

        option_tmp = options.get_option("compute_plot")
        self.compute_plot = False if option_tmp is None else option_tmp

        self.game = catcher.ContinuousCatcher(width=self.screen_size[0], height=self.screen_size[1])
        self.bar_size = self.game.bar.size
        self.fruit_size = self.game.fruit.size

        option_tmp = options.get_option("discrete")
        self.discrete = True if option_tmp is None else option_tmp

        if self.discrete:
            self.actions = [-self.screen_size[0] + i * self.fruit_size[0] for i in
                            range(0, 1 + (self.screen_size[0] // self.fruit_size[0]))]
            self.actions.extend([self.screen_size[0] - i * self.fruit_size[0] for i in
                                 range(0, 1 + (self.screen_size[0] // self.fruit_size[0]))])
            self.actions.append(0.0)

            self.actions.sort()
        else:
            self.discrete = False
            self.actions = [-self.screen_size[0], self.screen_size[0]]

        self.agent = agent
        if self.agent is not None:
            self.agent.init(self.actions, [self.screen_size, self.fruit_size, self.bar_size])
            option_tmp = options.get_option("online_feed")
            self.online_feed = True if option_tmp is None else option_tmp
        else:
            self.online_feed = False

        self.trainer_agent = options.get_option("trainer_agent")
        if self.trainer_agent is not None:
            self.trainer_agent.init(self.actions,
                                    [self.screen_size, self.fruit_size, self.bar_size])

    def process_event(self):
        """
        Process the events of pygame (allows to quit)
        Get the keyboard inputs when played with a player
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.user_left += 5
                elif event.key == pygame.K_RIGHT:
                    self.user_right += 5
                elif event.key == pygame.K_k:
                    self.running = False

    def draw(self, reward=float('nan')):
        """
        Draw the emulator to the screen

        :param reward: The total reward
        """

        self.surface.fill(self.color_style.background_color)  # Draw background

        fruit_center = self.game.fruit.center
        bar_center = self.game.bar.center
        fruit_size = self.fruit_size
        bar_size = self.bar_size

        pygame.draw.rect(self.surface, self.color_style.fruit_color,
                         ((fruit_center[0] - fruit_size[0] / 2.0,
                           fruit_center[1] - fruit_size[1] / 2.0),
                          (fruit_size[0], fruit_size[1])))  # Draw fruit
        pygame.draw.rect(self.surface, self.color_style.bar_color,
                         ((bar_center[0] - bar_size[0] / 2.0,
                           bar_center[1] - bar_size[1] / 2.0),
                          (bar_size[0], bar_size[1])))  # Draw bar

        # Draw the number of lives remaining
        font = pygame.font.SysFont('Arial', 24, True)
        text_string = 'Lives: {}    Reward: {:.3f}'.format(self.game.lives, reward)
        text = font.render(text_string, True, self.color_style.text_color,
                           self.color_style.background_color)
        text_rect = text.get_rect()
        text_size = font.size(text_string)
        text_rect.centerx = self.surface.get_rect()[0] + text_size[0] / 2.0
        text_rect.centery = self.surface.get_rect()[1] + text_size[1] / 2.0
        self.surface.blit(text, text_rect)

        if self.current_gen != 0:
            pygame.display.set_caption("Catcher: {}".format(self.current_gen))

        pygame.display.flip()
        pass

    def discretize(self, state):
        return (state[2] // self.fruit_size[0]) - (state[0] // self.fruit_size[0]),

    def reset(self):
        self.game.reset()

    def run(self, data_file = "test.csv"):
        """
        Emulate the catcher game
        """
        action = [0]
        state = self.game.observe()
        prev_state = self.game.observe()
        game_over = False

        total_reward = 0
        gamma = self.game.gamma()
        fps_counter = 0

        history = [0.0 for _ in range(0, self.max_generations)] if self.compute_plot else None
        start_gen_time = pygame.time.get_ticks()

        print("Starting emulator with agent",
              self.agent.name if self.agent is not None else "HumanAgent")
        print("Maximum number of generations is:",
              "infinite" if self.infinite_generations else self.max_generations)

        clock = pygame.time.Clock()
        start_time = pygame.time.get_ticks()
        
        #with open(data_file,"a") as test:
          #  test.write("bar_center_x, bar_velocity, fruit_center_x, fruit_center_y,reward,next_bar_center_x, next_bar_velocity, next_fruit_center_x, next_fruit_center_y\n")

        while self.running:
            if self.agent is None:
                action[0] = self.user_right - self.user_left
                self.user_right = 0
                self.user_left = 0
            else:
                if self.discrete:
                    discrete_state = self.discretize(state)

                    if self.trainer_agent is not None:
                        action[0] = self.trainer_agent.step(state)  # TODO TEST only, to be removed
                        # action[0] = self.trainer_agent.step(discrete_state)
                    else:
                        action[0] = self.agent.step(state)
                else:
                    action[0] = self.agent.step(state)
            state, reward, game_over = self.game.step(action)
            
            #Write necessary elements to a file for Fitted-Q to learn
            with open(data_file,"a") as test:
                for element in prev_state:
                    test.write(str(element) + ",")
                test.write(str(action[0]) + "," + str(reward))
                for element in state:
                    test.write("," + str(element))
                test.write("\n")
            

            if self.online_feed:
                if self.discrete:
                    discrete_state = self.discretize(prev_state)
                    self.agent.feed(discrete_state, action[0], reward)
                else:
                    self.agent.feed(prev_state, action[0], reward)

            prev_state = state

            total_reward += reward
            fps_counter += 1

            if self.display:
                self.draw(total_reward)

                self.process_event()

                # --- Maintain fps
                if math.isnan(self.fps) is False:
                    clock.tick(self.fps)

            if self.infinite_generations is False and self.max_frames != 0 and fps_counter >= self.max_frames:
                self.running = False

            if self.max_frames != 0 and fps_counter % (self.max_frames // 100) == 0:
                print("Frames:", fps_counter, "-", float(fps_counter) / (self.max_frames // 100),
                      ". Time:",
                      (pygame.time.get_ticks() - start_gen_time) / 1000.0)

            if game_over:
                # Lost game
                if self.compute_plot:
                    history[self.current_gen] = total_reward
                if self.current_gen % self.display_generations_gap == 0:
                    print("Processing:", self.current_gen, "time played:",
                          (pygame.time.get_ticks() - start_gen_time) / 1000.0)
                    start_gen_time = pygame.time.get_ticks()

                if self.infinite_generations is False and self.current_gen >= self.max_generations - 1:
                    self.running = False

                self.game.reset()
                self.current_gen += 1
                total_reward = 0

        if self.compute_plot:
            cum_sum_plot = np.cumsum(history)
            cum_sum_plot = cum_sum_plot / np.arange(1.0, cum_sum_plot.size + 1, 1.0)
            plt.plot(history)
            plt.plot(cum_sum_plot)
            plt.show()

        print("Time played:", (pygame.time.get_ticks() - start_time) / 1000.0)

        pygame.display.quit()
        pass

    @staticmethod
    def emulate(agent_emulate, options):
        """
        Fully emulate the game

        :param agent_emulate: The agent to use (None for user controlled)
        :param options: Options used by the emulator
        """
        Emulator(options, agent_emulate).run()


class EmulatorOptions:
    def __init__(self, verbose=True):
        self.verbosity = verbose if (verbose is True or verbose is False) else True
        self.options = dict()

        # Setting default values
        self.options["show_display"] = True
        self.options["screen_size"] = None
        self.options["fps"] = 30
        self.options["color_style"] = DefaultColorStyle()
        self.options["discrete"] = True
        self.options["online_feed"] = True
        self.options["max_generations"] = 1000
        self.options["display_generations_gap"] = 250
        self.options["infinite_generations"] = False
        self.options["compute_plot"] = False
        self.options["trainer_agent"] = None
        self.options["max_frames"] = 0

        pass

    def add_option(self, key, value):
        if key in self.options:
            if self.verbosity:
                print("Overriding entry:", key, "which had value:", self.options[key])
            self.options[key] = value
        else:
            self.options[key] = value
        pass

    def get_option(self, key):
        if key in self.options:
            return self.options[key]
        else:
            return None


if __name__ == "__main__":

    
    discretized_agent = TreeAgent("big_tree.sav")
    discretizedAgent_options = EmulatorOptions(False)
    discretizedAgent_options.add_option("show_display", True)
    discretizedAgent_options.add_option("online_feed", False)
    discretizedAgent_options.add_option("compute_plot", False)
    discretizedAgent_options.add_option("max_frames", 0)
    discretizedAgent_options.add_option("infinite_generations", True)
    discretizedAgent_options.add_option("trainer_agent", None)

    try:
        Emulator.emulate(discretized_agent, discretizedAgent_options)
    except KeyboardInterrupt:
        print("Interrupted after visualisation")
    # discretized_agent.save("saves/visualized")  # Useless, is not online_feed here
