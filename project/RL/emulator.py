import numpy as np
import pygame

import BaseAgent
import catcher


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
    def __init__(self, show_display=True, screen_size=None, agent=None, color_style=None):
        """
        Create the emulator with all the necessary variables

        :param show_display: Whether to show a display or not
        :param screen_size: The size of the screen
        :param agent: The agent to use (None for user controlled)
        :param color_style: The color style to use
        """
        if screen_size is None:
            screen_size = [400, 400]
        if color_style is None:
            self.color_style = DefaultColorStyle()

        self.running = True
        self.user_right = 0
        self.user_left = 0

        self.display = show_display
        self.screen_size = screen_size
        if show_display is True:
            pygame.display.init()
            pygame.font.init()
            self.surface = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Catcher")

        self.game = catcher.ContinuousCatcher(width=self.screen_size[0], height=self.screen_size[1])
        self.bar_size = self.game.bar.size
        self.fruit_size = self.game.fruit.size

        self.agent = agent
        if self.agent is not None:
            self.agent.init([self.screen_size, self.fruit_size, self.bar_size])

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

        pygame.display.flip()
        pass

    def reset(self):
        self.game.reset()

    def run(self):
        """
        Emulate the catcher game
        """
        action = [0]
        state = self.game.observe()
        game_over = False

        total_reward = 0
        gamma = self.game.gamma()
        counter = 0
        fps_counter = 0

        clock = pygame.time.Clock()
        start_time = pygame.time.get_ticks()

        while self.running:
            if self.agent is None:
                action[0] = self.user_right - self.user_left
                self.user_right = 0
                self.user_left = 0
            else:
                action[0] = self.agent.step(state)
            state, reward, game_over = self.game.step(action)

            total_reward += (np.power(gamma, counter) * reward)
            if fps_counter % self.game.fps == 0:
                counter += 1

            fps_counter += 1

            if self.display:
                self.draw(total_reward)

                self.process_event()

                # --- Maintain fps
                clock.tick(self.game.fps)

            if game_over:
                print('You loose!')
                self.running = False

        print("Time played:", (pygame.time.get_ticks() - start_time) / 1000.0)

        pygame.display.quit()
        pass

    @staticmethod
    def emulate(agent, show_display=True, screen_size=None, color_style=None):
        """
        Fully emulate the game

        :param agent: The agent to use (None for user controlled)
        :param show_display: Whether to show a display or not
        :param screen_size: The size of the screen
        :param color_style: The color style to use
        """
        Emulator(show_display=show_display, screen_size=screen_size, agent=agent,
                 color_style=color_style).run()


if __name__ == "__main__":
    Emulator.emulate(BaseAgent.BaseAgent())
    # Emulator(agent=BaseAgent.BaseAgent()).run()
