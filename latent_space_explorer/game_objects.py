from .utils import random_circle_init

import math
import pygame

class Point:
    # Point representing a prompt/generation
    """
    :param text: Text associated with the point
    :param encoding: Encoding associated with the point
    :param xy_pos: Tuple (x, y) for the position of point in R2. If not given, will initialize randomly
    :param xy_init_kwargs: kwargs to random_circle_init for randomly init'ing xy_pos
    """
    def __init__(self, text, encoding, xy_pos = None, xy_init_kwargs = {}):
        self.text = text

        if xy_pos is not None:
            self.xy_pos = xy_pos # Tuple of x and y in R2 space (not screen space)
        else:
            self.xy_pos = random_circle_init(**xy_init_kwargs)
        self.encoding = encoding

        if "on_edge" in xy_init_kwargs:
            self.on_edge = xy_init_kwargs['on_edge']
        else:
            self.on_edge = False

    def move(self, new_xy_pos):
        if self.on_edge:
            x, y = new_xy_pos
            length = math.sqrt(x**2 + y**2)
            self.xy_pos = (x/length, y/length)
        else:
            self.xy_pos = new_xy_pos

class TextPrompt:
    def __init__(self, prompt_text, font, screen):
        self.prompt_text = prompt_text
        self.font = font
        self.screen = screen

        self.user_input = ""

    def draw_main_blocks(self):
        """
        Draw main text block and text inside it
        """
        screen_width, screen_height = pygame.display.get_surface().get_size()

        # Margins
        rect_height_fraction = 0.3
        rect_width_fraction = 0.7
        text_prompt_fraction = 0.3
        border_thickness = 10

        rect_height = int(screen_height * rect_height_fraction)
        rect_width = int(screen_width * rect_width_fraction)

        rect_x = (screen_width - rect_width) // 2
        rect_y = (screen_height - rect_height) // 2

        rect = pygame.Rect(rect_x, rect_y, rect_width, rect_height)

        pygame.draw.rect(self.screen, (0, 0, 0), rect)  # Fill rectangle with black
        pygame.draw.rect(self.screen, (255, 255, 255), rect, border_thickness)  # Draw white border

        text_surface = self.font.render(self.prompt_text, True, (255, 255, 255))  # Render text
        text_rect = text_surface.get_rect()  # Get text rectangle
        text_rect.centerx = rect.centerx  # Center text horizontally
        text_rect.y = rect.y + int(rect.height * text_prompt_fraction)  # Position text vertically based on fraction
        self.screen.blit(text_surface, text_rect)  # Draw text

        user_text_surface = self.font.render(self.user_input, True, (255, 255,255))
        user_text_rect = user_text_surface.get_rect()
        user_text_rect.centerx = rect.centerx
        user_text_rect.y = rect.y + int(rect.height * (1 - text_prompt_fraction))
        self.screen.blit(user_text_surface, user_text_rect)

    def get_user_input(self) -> bool:
        """
        Get user input, update self.user_input, and return True if user pressed enter
        """
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_BACKSPACE:
                    self.user_input = self.user_input[:-1]
                elif event.key == pygame.K_RETURN:
                    return True
                else:
                    self.user_input += event.unicode
        return False

    def update(self):
        is_done = self.get_user_input()
        self.draw_main_blocks()

        return is_done
