from typing import List

from dataclasses import dataclass
from .fast_sd import fast_diffusion_pipeline

import torch
import pygame
import numpy as np
import time
from PIL import Image

from .game_objects import Point, TextPrompt
from .sampling import (
    DistanceSampling,
    CircleSampling
)

@dataclass
class GameConfig:
    point_thickness : float = 10 # Thickness for each point
    zoom_speed : float = 0.75 # How fast we zoom in or out
    move_speed : float = 0.75 # How fast we move around canvas
    point_font_size : int = 25 # Size of fonts for points on screen

    prompt_font_size : int = 30 # Size of font for prompt on screen

    # screen size
    width : int = 1920
    height : int = 1080

    # size of sample in top left
    sample_width : int = 512
    sample_height : int = 512

    compile : bool = False # compile the sd model with torch.compile?
    sampler : str = "distance" # "distance" or "circle"
    seed : int = 0 # Seed for initial latent noise
    call_every : int = 90 # Only calls draw function every *this many* ms. This is to prevent lag. Set this to be around the latency of the model

class LatentSpaceExplorer:
    def __init__(self, config : GameConfig = GameConfig()):
        self.config = config

        self.pipe = fast_diffusion_pipeline(compile = self.config.compile)
        self.points : List[Point] = []
        self.player_pos = None # [2,] np array in R2 space

        self.dragging_point_idx = None
        self.selected_point_idx = None

        self.zoom_level = 300.0
        self.translation = np.array([-self.config.width/2, -self.config.height/2])

        self.point_kwargs = {}
        if self.config.sampler == "distance":
            self.sampler = DistanceSampling
        elif self.config.sampler == "circle":
            self.sampler = CircleSampling
            self.point_kwargs['on_edge'] = True
        else:
            raise ValueError(f"Invalid sampler choice: {self.config.sampler}")

        pygame.init()
        self.screen = pygame.display.set_mode((self.config.width, self.config.height))
        self.clock = pygame.time.Clock()
        self.ms_elapsed = 0

        # (n_samples, running average)
        self.avg_latency = (0, 0) # Track average latency of generation for debug
        
        self.sample_image = None
        self.sample_font = pygame.font.Font(None, self.config.point_font_size)
        
        # User input
        self.input_font = pygame.font.Font(None, self.config.prompt_font_size)
        self.inputting_text = False
        self.inputting_text_for = None # oneof ["modify", "add"]
        self.text_prompt : TextPrompt = None
    
    def tick(self):
        self.clock.tick()
        self.ms_elapsed += self.clock.get_time()

    def update_latency(self, new_observation):
        n = self.avg_latency[0]
        old_avg = self.avg_latency[1]
        self.avg_latency = (n + 1, (old_avg * n + new_observation) / (n + 1))

    def create_text_prompt(self, prompt_text):
        self.text_prompt = TextPrompt(prompt_text, self.input_font, self.screen)

    def switch_sampler(self):
        if self.config.sampler == "distance":
            self.config.sampler = "circle"
            self.sampler = CircleSampling
            self.point_kwargs = {'on_edge' : True}
        elif self.config.sampler == "circle":
            self.config.sampler = "distance"
            self.sampler = DistanceSampling
            self.point_kwargs = {}
        self.set_prompts(self.prompts, reset = True)
    
    @property
    def encodes(self):
        """
        Get encodings directly from points as a tuple with batched encodings
        """
        if not self.points:
            return None
        encode_list = [p.encoding for p in self.points] # list of N-tuples
        n = len(encode_list[0])
        res = []
        for i in range(n):
            res.append(torch.cat([e[i] for e in encode_list], dim = 0) if encode_list[0][i] is not None else None)

        return tuple(res)
    
    @property
    def prompts(self):
        """
        Get a list of current prompts
        """
        return [p.text for p in self.points]
    
    @property
    def r2_points(self):
        """
        Get all points in terms of R2 space
        """
        points = [np.array(p.xy_pos) for p in self.points]
        points = np.stack(points, axis = 0) # [n, 2]
        return points

    @property
    def screen_space_points(self):
        """
        Get all points in terms of screen space
        """
        screen_space = (self.r2_points * self.zoom_level) - self.translation[None,:]
        return screen_space # list of points in scren space
    
    @property
    def mouse_pos(self):
        return np.array(pygame.mouse.get_pos())

    def invert_screen_space(self, point):
        """
        taking position as [2,] np array in screen space, return R2 pos
        """
        return (point + self.translation) / self.zoom_level

    def screen_space(self, point):
        """
        R2 -> screenspace as [2,] array
        """
        return (point * self.zoom_level) - self.translation
    
    def fixed_seed(self):
        """
        Controls random number generator for initial latent noise
        """
        return torch.Generator('cuda').manual_seed(self.config.seed)
    
    def get_encodes(self, text):
        """
        Get text encodings for some prompt then split them so we can associate points with thier encodings
        """
        encodes = self.pipe.get_encodes(text, generator = self.fixed_seed())
        # (n-tuple of lists) into (list of n-tuples)
        if not isinstance(encodes, tuple) and not isinstance(encodes, list):
            return encodes # Already a tensor, no problem
        
        res_list = []
        for i in range(len(encodes[0])):
            res_list_i = [encodes_j[i].unsqueeze(0) if encodes_j is not None else None for encodes_j in encodes]
            res_list.append(tuple(res_list_i))

        return res_list
    
    def draw_sample(self):
        """
        Draw sample with current points and player position
        """
        if self.player_pos is not None and self.encodes is not None:
            if self.ms_elapsed >= self.config.call_every:
                time_start = time.time()
                encoding = self.sampler(self.encodes)(self.player_pos, self.r2_points)
                self.sample_image = self.pipe.generate_from_encodes(encoding, generator = self.fixed_seed()).images[0]
                time_total = float(time.time() - time_start) * 1000 # s -> ms

                self.update_latency(time_total)

                self.ms_elapsed = 0

    def get_player_pos_r2(self):
        """
        Get player position in R2 from the 
        """
        self.player_pos = self.invert_screen_space(self.mouse_pos)

    def get_player_pos_screenspace(self):
        """
        Get player pos in screen space
        """
        if self.player_pos is not None: return self.screen_space(self.player_pos)
    
    def detect_mouse_on_point(self):
        """
        Detect if mouse is currently in a point. If so, returns index of point, otherwise returns none.
        """
        if not self.points:
            return None
        
        mouse_pos = self.mouse_pos
        points = self.screen_space_points

        distances = np.linalg.norm(points - mouse_pos[None,:], axis = 1)
        close_idx = np.argmin(distances)

        if distances[close_idx] <= self.config.point_thickness:
            return close_idx
        return None

    # === POINT/NODE CONTROL ===

    def modify_node(self, new_prompt):
        idx = self.selected_point_idx
        new_prompts = self.prompts
        new_prompts[idx] = new_prompt
        self.set_prompts(new_prompts, reset = False)
    
    def add_node(self, new_prompt):
        self.set_prompts(self.prompts + [new_prompt], reset = False)

    def del_node(self):
        idx = self.selected_point_idx
        new_prompts = list(self.prompts)
        del new_prompts[idx]
        self.set_prompts(new_prompts, reset = False)
        self.selected_point_idx = None

    def prepare_to_prompt(self, mode):
        """
        Get ready to show the textbox. Call when we want the text prompt to come
        """
        self.inputting_text = True
        self.inputting_text_for = mode

        if mode == "modify":
            self.create_text_prompt("Enter New Prompt To Replace Node:")
        elif mode == "add":
            self.create_text_prompt("Enter New Prompt To Create Node:")
    
    def handle_prompt(self):
        """
        After enter pressed with textbox, this is called to go back to normal game
        """
        done_prompting = self.text_prompt.update()

        if done_prompting:
            new_prompt = self.text_prompt.user_input.strip()
            if self.inputting_text_for == "modify":
                self.modify_node(new_prompt)
            elif self.inputting_text_for == "add":
                self.add_node(new_prompt)
            self.text_prompt = None
            self.inputting_text = False

    def set_prompts(self, prompts : List[str], reset : bool = False):
        """
        :param prompts: New prompts to update to
        :param reset: Reset xy positions of points?
        """

        if len(prompts) > 0:
            encodes = self.get_encodes(prompts)

        # First call
        if not self.points or reset:
            self.points = [Point(prompt, encoding, xy_init_kwargs = self.point_kwargs) for (prompt, encoding) in zip(prompts, encodes)]
            return
    
        # Modifications
        old_len = len(self.points)
        new_len = len(prompts)
    
        pos = [tuple(pos_i) for pos_i in self.r2_points] # positions for each point

        if old_len <= new_len: # Additions or modification
            pos += [None] * (new_len - old_len) # randomly init this many new positions
            self.points = [Point(prompt, encoding, pos_i, xy_init_kwargs = self.point_kwargs) for (prompt, encoding, pos_i) in zip(prompts, encodes, pos)]
            return
        elif old_len > new_len: # Deletions
            idx_to_keep = []
            for idx, prompt in enumerate(self.prompts):
                if prompt in prompts:
                    idx_to_keep.append(idx)
            self.points = [self.points[idx] for idx in idx_to_keep]
            return

    # === CONTROLS ===

    def handle_event_controls(self):
        """
        Handles discrete (i.e. keydown, mousedown) controls through events
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Click
                if event.button == 1: # Left click
                    self.selected_point_idx = self.detect_mouse_on_point()
                    if self.selected_point_idx is not None: self.dragging_point_idx = None
                    else: # If no point was selected, we move player cursor
                        self.get_player_pos_r2()
                        self.draw_sample()
                elif event.button == 3: # Right click
                    self.dragging_point_idx = self.detect_mouse_on_point()
                    if self.dragging_point_idx is not None: self.selected_point_idx = None
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3: # Right Up
                    self.dragging_point_idx = None # Disable drag
            elif event.type == pygame.MOUSEMOTION:
                if self.dragging_point_idx is not None:
                    # Drag point
                    self.points[self.dragging_point_idx].move(self.invert_screen_space(self.mouse_pos))
                elif pygame.mouse.get_pressed()[0]:
                    self.get_player_pos_r2()
                    self.draw_sample()
            elif event.type == pygame.KEYDOWN:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_r]:
                    self.set_prompts(self.prompts, reset = True)
                elif keys[pygame.K_t] and self.selected_point_idx is not None: # Modify existing node
                    self.prepare_to_prompt("modify")
                    return
                elif keys[pygame.K_p]: # Adding a node
                    self.prepare_to_prompt("add")
                    return
                elif keys[pygame.K_o] and self.selected_point_idx is not None:
                    # Remove node
                    self.del_node()
                elif keys[pygame.K_g]:
                    if self.sample_image is not None:
                        self.sample_image.save("sample.png")
                elif keys[pygame.K_m]:
                    # Change sampler mode
                    self.switch_sampler()


    def handle_continuous_controls(self):
        """
        Continuous controls for movement (i.e. zoom, movement)
        """
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            self.zoom_level = max(0.01, self.zoom_level - self.config.zoom_speed)
        elif keys[pygame.K_e]:
            self.zoom_level = self.zoom_level + self.config.zoom_speed

        idx, sign = None, None

        # up, down, left, right
        if keys[pygame.K_w]:
            idx, sign = 1, -1
        elif keys[pygame.K_s]:
            idx, sign = 1, 1
        elif keys[pygame.K_a]:
            idx, sign = 0, -1
        elif keys[pygame.K_d]:
            idx, sigh = 0, 1

        if idx is not None and sign is not None:
            self.translation[idx] += sign * self.config.move_speed

    # === DRAWING THINGS ===

    def draw_main_screen(self):
        """
        Draw main screen. Sample image, points, etc.
        """
        def get_point_color(idx):
            color = (255, 255, 255) # default to white
            if idx == self.selected_point_idx:
                color = (0, 127.5, 0)
            if idx == self.dragging_point_idx:
                color = (255, 0, 0)
            return color
        
        if self.config.sampler == "circle":
            # Draw unit circle on screen
            center = np.array([0,0])
            border = np.array([1,0])

            center = self.screen_space(center)
            border = self.screen_space(border)
            radius = abs(border[0] - center[0])

            pygame.draw.circle(self.screen, (255, 255, 255), center, int(radius), 1)
        
        if len(self.points) > 0:
            for idx, point in enumerate(self.screen_space_points):
                pygame.draw.circle(self.screen, get_point_color(idx), point, self.config.point_thickness)
                text = self.sample_font.render(self.points[idx].text, True, get_point_color(idx))
                self.screen.blit(text, point)
        
        player_pos = self.get_player_pos_screenspace()
        if player_pos is not None:
            pygame.draw.circle(self.screen, (0, 255, 0), player_pos, self.config.point_thickness/2)
        
        if self.sample_image is not None:
            pygame_image = pygame.image.fromstring(self.sample_image.tobytes(), self.sample_image.size, self.sample_image.mode)
            pygame_image = pygame.transform.scale(pygame_image, (self.config.sample_width, self.config.sample_height))
            self.screen.blit(pygame_image, (0, 0))

    def update(self):
        """
        Main pygame loop
        """

        if not self.inputting_text:
            self.handle_event_controls()
            self.handle_continuous_controls()
        self.tick()
        
        self.screen.fill((0,0,0))
        self.draw_main_screen()

        # Handle prompt after so it can be drawn over the main screen
        if self.inputting_text:
            self.handle_prompt()
        pygame.display.flip()
