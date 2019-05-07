import numpy as np
import pygame

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


class Display:
	def __init__(self, screen_size=None, color_style=None):
		if screen_size is None:
			screen_size = [400, 400]
		if color_style is None:
			self.color_style = DefaultColorStyle()
			
		self.running = True
		self.screen_size = screen_size
		pygame.display.init()
		pygame.font.init()
		self.surface = pygame.display.set_mode(self.screen_size)
		pygame.display.set_caption("Catcher")
		
		self.game = catcher.ContinuousCatcher(width=self.screen_size[0], height=self.screen_size[1])
		self.bar_size = self.game.bar.size
		self.fruit_size = self.game.fruit.size
		
	def process_event(self):
		"""
		Process the events of pygame (allows to quit)
		Get the keyboard inputs when played with a player
		"""
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				pygame.display.quit()
			elif event.type == pygame.KEYDOWN:
				if event.key == pygame.K_LEFT:
					self.user_left += 5
				elif event.key == pygame.K_RIGHT:
					self.user_right += 5
		
	def draw(self, bar_center, fruit_center, reward_sum, lives):
		"""
		Draw to the screen

		state: state of the game to draw
		reward_sum: The total reward
		lives: lives of the game to draw
		"""
		# clock = pygame.time.Clock()
		self.surface.fill(self.color_style.background_color)  # Draw background

		fruit_center = fruit_center
		bar_center = bar_center
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
		text_string = 'Lives: {}    Reward: {:.3f}'.format(lives, reward_sum)
		text = font.render(text_string, True, self.color_style.text_color,
						   self.color_style.background_color)
		text_rect = text.get_rect()
		text_size = font.size(text_string)
		text_rect.centerx = self.surface.get_rect()[0] + text_size[0] / 2.0
		text_rect.centery = self.surface.get_rect()[1] + text_size[1] / 2.0
		self.surface.blit(text, text_rect)

		pygame.display.flip()
		self.process_event()

		# # --- Maintain fps
		# clock.tick(self.game.fps)
		