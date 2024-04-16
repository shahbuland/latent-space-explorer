from typing import List
import torch
import pygame
import numpy as np
import random
from fast_sd import fast_diffusion_pipeline

POINT_THICKNESS = 10
ZOOM_SPEED = 0.75
MOVE_SPEED = 0.75
FONT_SIZE = 25
WIDTH, HEIGHT = 1920, 1080
SAMPLE_WIDTH, SAMPLE_HEIGHT = 512, 512

class LatentSpaceExplorer:
    """
    PyGame-based interactive space explorer. Prompts as points are rendered
    on the screen and user can click to place themselves anywhere in the space.
    """
    def __init__(self, compile=False):
        self.pipe = fast_diffusion_pipeline(compile=compile)
        self.prompts = []
        self.stacked_points = None
        self.player_x = None
        self.player_y = None
        self.encodes = None
        self.zoom_level = 100.0
        self.translation = np.array([-WIDTH / 2, -HEIGHT / 2])
        self.dragging_point_index = None

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.font = pygame.font.Font(None, FONT_SIZE)
        self.clock = pygame.time.Clock()
        self.sample_image = None

    def fixed_seed(self, seed: int = 0):
        return torch.Generator('cuda').manual_seed(0)

    def transform_points(self):
        return (self.stacked_points * self.zoom_level) - self.translation

    def inverse_transform_point(self, point):
        return (point + self.translation) / self.zoom_level

    def draw_sample(self):
        z = np.array([self.player_x, self.player_y])
        x = self.stacked_points
        p = 2
        coefs = 1. / ((1. + np.linalg.norm(z[None, :] - x, axis=1) ** p))
        print(coefs)
        if self.encodes is not None:
            coefs = torch.from_numpy(coefs).to('cuda').half()
            encode_list = []
            for encode in self.encodes:
                if encode is None:
                    encode_list.append(None)
                else:
                    encode_list.append(
                        (coefs[:, None, None] * encode).sum(0) if len(encode.shape) == 3 else (coefs[:, None] * encode).sum(0)
                    )
            res = self.pipe.generate_from_encodes(encode_list, generator=self.fixed_seed()).images[0]
            return res
        
        return None

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    self.player_x, self.player_y = pygame.mouse.get_pos()
                elif event.button == 3:  # Right mouse button
                    mouse_pos = np.array(pygame.mouse.get_pos())
                    transformed_points = self.transform_points()
                    distances = np.linalg.norm(transformed_points - mouse_pos, axis=1)
                    closest_point_index = np.argmin(distances)
                    if distances[closest_point_index] <= POINT_THICKNESS:
                        self.dragging_point_index = closest_point_index
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:  # Right mouse button
                    self.dragging_point_index = None
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_point_index is not None:
                    self.stacked_points[self.dragging_point_index] = self.inverse_transform_point(np.array(pygame.mouse.get_pos()))
                elif pygame.mouse.get_pressed()[0]:  # Left mouse button
                    if self.player_x is not None and self.player_y is not None:
                        self.player_x, self.player_y = self.inverse_transform_point(np.array(pygame.mouse.get_pos()))
                        self.sample_image = self.draw_sample()
            elif event.type == pygame.KEYDOWN:
                if pygame.key.get_pressed()[pygame.K_r]:
                    self.set_prompts(self.prompts)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.zoom_level = max(0.01, self.zoom_level - ZOOM_SPEED)
        if keys[pygame.K_e]:
            self.zoom_level = self.zoom_level + ZOOM_SPEED
        if keys[pygame.K_w]:
            self.translation[1] -= MOVE_SPEED
        if keys[pygame.K_s]:
            self.translation[1] += MOVE_SPEED
        if keys[pygame.K_a]:
            self.translation[0] -= MOVE_SPEED
        if keys[pygame.K_d]:
            self.translation[0] += MOVE_SPEED

        self.clock.tick()
        self.screen.fill((0, 0, 0))

        for index, (point, prompt) in enumerate(zip(self.transform_points(), self.prompts)):
            color = (255, 0, 0) if index == self.dragging_point_index else (255, 255, 255)
            pygame.draw.circle(self.screen, color, (int(point[0]), int(point[1])), POINT_THICKNESS)
            text = self.font.render(prompt, True, color)
            self.screen.blit(text, (int(point[0]), int(point[1])))

        if self.player_x is not None and self.player_y is not None:
            player_x = (self.player_x * self.zoom_level) - self.translation[0,]
            player_y = (self.player_y * self.zoom_level) - self.translation[1,]
            pygame.draw.circle(self.screen, (0, 255, 0), (player_x, player_y), POINT_THICKNESS // 2)

        if self.sample_image is not None:
            pygame_image = pygame.image.fromstring(self.sample_image.tobytes(), self.sample_image.size, self.sample_image.mode)
            pygame_image = pygame.transform.scale(pygame_image, (SAMPLE_WIDTH, SAMPLE_HEIGHT))
            self.screen.blit(pygame_image, (0, 0))

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
        self.encodes = self.pipe.get_encodes(prompts, generator = self.fixed_seed())

if __name__ == "__main__":
    explorer = LatentSpaceExplorer()
    explorer.set_prompts(
        [
            "A photo of a cat",
            "A space-aged ferrari",
            "artwork of the titanic hitting an iceberg",
            "a photo of a dog"
        ]
    )

    while True:
        explorer.update()
