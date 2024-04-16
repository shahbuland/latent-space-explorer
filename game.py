from typing import List

import torch
import pygame
import numpy as np
import random

POINT_THICKNESS = 10
ZOOM_SPEED = 0.01
MOVE_SPEED = 0.01

class LatentSpaceExplorer:
    """
    PyGame-based interactive space explorer. Prompts as points are rendered
    on the screen and user can click to place themselves anywhere in the space.
    """
    def __init__(self, compile = False):
        self.prompts = []
        self.stacked_points = None
        self.player_x = None
        self.player_y = None
        self.encodes = None
        self.zoom_level = 1.5
        self.translation = np.array([-400, -300])
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        self.font = pygame.font.Font(None, 36)

    def fixed_seed(self, seed : int = 0):
        return torch.Generator('cuda').manual_seed(0)

    def transform_points(self):
        return (self.stacked_points * self.zoom_level * 266.67) - self.translation
    
    def inverse_transform_point(self, point):
        return (point + self.translation) / (self.zoom_level * 266.67)

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                self.player_x, self.player_y = pygame.mouse.get_pos()
            elif event.type == pygame.MOUSEMOTION and (pygame.mouse.get_pressed()[0] or pygame.key.get_pressed()[pygame.K_SPACE]):
                if self.player_x is not None and self.player_y is not None:
                    self.player_x, self.player_y = pygame.mouse.get_pos()
            keys = pygame.key.get_pressed()
            if event.type == pygame.KEYDOWN:
                if keys[pygame.K_r]:
                    self.set_prompts(self.prompts)
            if keys[pygame.K_q]:
                self.zoom_level += ZOOM_SPEED
            if keys[pygame.K_e]:
                self.zoom_level -= ZOOM_SPEED
            if keys[pygame.K_w]:
                self.translation[1] -= MOVE_SPEED * 266.67
            if keys[pygame.K_s]:
                self.translation[1] += MOVE_SPEED * 266.67
            if keys[pygame.K_a]:
                self.translation[0] -= MOVE_SPEED * 266.67
            if keys[pygame.K_d]:
                self.translation[0] += MOVE_SPEED * 266.67

        self.screen.fill((0, 0, 0))
        for point, prompt in zip(self.transform_points(), self.prompts):
            pygame.draw.circle(self.screen, (255, 255, 255), (int(point[0]), int(point[1])), POINT_THICKNESS)
            text = self.font.render(prompt, True, (255, 255, 255))
            self.screen.blit(text, (int(point[0]), int(point[1])))

        if self.player_x is not None and self.player_y is not None:
            pygame.draw.circle(self.screen, (0, 255, 0), (self.player_x, self.player_y), POINT_THICKNESS//2)

        pygame.display.flip()

    def set_prompts(self, prompts : List[str]):
        """
        Set the internal cache of prompts
        """
        assert len(prompts) > 1, "Need more than one prompt to explore"

        def disjoint_uniform(a, b, c, d):
            if random.random() <= 0.5:
                return random.uniform(a, b)
            else:
                return random.uniform(c, d)

        # For visualization
        self.prompts = prompts
        x_values = np.linspace(-1, 1, len(prompts))
        y_values = np.array([0 if i == 0 or i == len(x_values) - 1 else disjoint_uniform(-0.75, -0.25,0.25,0.75) for i in range(len(x_values))])

        self.stacked_points = np.stack([x_values, y_values], axis = -1)
if __name__ == "__main__":
    explorer = LatentSpaceExplorer()
    explorer.set_prompts(
        ["A photo of a cat", "A space-aged ferrari", "artwork of the titanic hitting an iceberg", "a photo of a dog"]
    )

    while True:
        explorer.update()
