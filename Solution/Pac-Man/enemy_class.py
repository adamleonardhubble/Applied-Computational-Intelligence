import pygame
import random
import math
from settings import *

vec = pygame.math.Vector2


class Enemy:
    # Initialize Enemy Character
    def __init__(self, app, pos, input_screen, EnemyIndex):
        # Initalize Global Enemy Info
        self.app = app
        self.grid_pos = pos
        self.starting_pos = [pos.x, pos.y]
        self.pix_pos = self.get_pix_pos()
        self.radius = int(self.app.cell_width//2.3)
        self.direction = vec(0, 0)
        self.target = None
        self.display_screen = input_screen

        # Define Animation Variables
        self.SpriteSheetFrameCounter = 0
        self.SpriteSheetFrameOffset = 0

        # Assign Individual Ghost Name, Colour and Personality
        if(EnemyIndex == 0):
            # Init Self
            self.name = 'Blinky_(Red)'
            self.colour = BLINKY_RED_COLOUR
            self.personality = "speedy"
            self.speed = 2

            # Load Red Ghost Sprite Sheet Images
            self.GhostSpriteSheet = [pygame.image.load('Ghost_Red_Down_1.png'), pygame.image.load('Ghost_Red_Down_2.png'), pygame.image.load('Ghost_Red_Left_1.png'), pygame.image.load('Ghost_Red_Left_2.png'), pygame.image.load('Ghost_Red_Right_1.png'), pygame.image.load('Ghost_Red_Right_2.png'), pygame.image.load('Ghost_Red_Up_1.png'), pygame.image.load('Ghost_Red_Up_2.png')]
        if(EnemyIndex == 1):
            # Init Self
            self.name = 'Pinky_(Pink)'
            self.colour = PINKY_PINK_COLOUR
            self.personality = "slow"
            self.speed = 1

            # Load Pink Ghost Sprite Sheet Images
            self.GhostSpriteSheet = [pygame.image.load('Ghost_Pink_Down_1.png'), pygame.image.load('Ghost_Pink_Down_2.png'), pygame.image.load('Ghost_Pink_Left_1.png'), pygame.image.load('Ghost_Pink_Left_2.png'), pygame.image.load('Ghost_Pink_Right_1.png'), pygame.image.load('Ghost_Pink_Right_2.png'), pygame.image.load('Ghost_Pink_Up_1.png'), pygame.image.load('Ghost_Pink_Up_2.png')]
        if(EnemyIndex == 2):
            # Init Self
            self.name = 'Inky_(Blue)'
            self.colour = INKY_BLUE_COLOUR
            self.personality = "random"
            self.speed = 1

            # Load Blue Ghost Sprite Sheet Images
            self.GhostSpriteSheet = [pygame.image.load('Ghost_Blue_Down_1.png'), pygame.image.load('Ghost_Blue_Down_2.png'), pygame.image.load('Ghost_Blue_Left_1.png'), pygame.image.load('Ghost_Blue_Left_2.png'), pygame.image.load('Ghost_Blue_Right_1.png'), pygame.image.load('Ghost_Blue_Right_2.png'), pygame.image.load('Ghost_Blue_Up_1.png'), pygame.image.load('Ghost_Blue_Up_2.png')]
        if(EnemyIndex == 3):
            # Init Self
            self.name = 'Clyde_(Orange)'
            self.colour = CLYDE_ORANGE_COLOUR
            self.personality = "scared"
            self.speed = 2

            # Load Orange Ghost Sprite Sheet Images
            self.GhostSpriteSheet = [pygame.image.load('Ghost_Orange_Down_1.png'), pygame.image.load('Ghost_Orange_Down_2.png'), pygame.image.load('Ghost_Orange_Left_1.png'), pygame.image.load('Ghost_Orange_Left_2.png'), pygame.image.load('Ghost_Orange_Right_1.png'), pygame.image.load('Ghost_Orange_Right_2.png'), pygame.image.load('Ghost_Orange_Up_1.png'), pygame.image.load('Ghost_Orange_Up_2.png')]
    
    # Update Enemy
    def update(self):
        self.target = self.set_target()
        if self.target != self.grid_pos:
            self.pix_pos += self.direction * self.speed
            if self.time_to_move():
                self.move()

        # Setting grid position in reference to pix position
        self.grid_pos[0] = (self.pix_pos[0]-TOP_BOTTOM_BUFFER +
                            self.app.cell_width//2)//self.app.cell_width+1
        self.grid_pos[1] = (self.pix_pos[1]-TOP_BOTTOM_BUFFER +
                            self.app.cell_height//2)//self.app.cell_height+1

    def draw(self):
        # Set Correct Frame Offset
        if(self.direction == vec(0, 1)): # If Moving Down
            self.SpriteSheetFrameOffset = 0
        if(self.direction == vec(-1, 0)): # If Moving Left
            self.SpriteSheetFrameOffset = 2
        if(self.direction == vec(1, 0)): # If Moving Right
            self.SpriteSheetFrameOffset = 4
        if(self.direction == vec(0, -1)): # If Moving Up
            self.SpriteSheetFrameOffset = 6
        
        # Draw Self
        self.display_screen.blit(self.GhostSpriteSheet[self.SpriteSheetFrameOffset + math.floor(self.SpriteSheetFrameCounter)], (self.pix_pos.x - 8, self.pix_pos.y - 8))

        # Increment SpriteSheetImageCounter
        self.SpriteSheetFrameCounter = self.SpriteSheetFrameCounter + 0.2
        if(self.SpriteSheetFrameCounter >= 2):
            self.SpriteSheetFrameCounter = 0

    def set_target(self):
        if self.personality == "speedy" or self.personality == "slow":
            return self.app.player.grid_pos
        else: # If Random or Scared
            if self.app.player.grid_pos[0] > COLS//2 and self.app.player.grid_pos[1] > ROWS//2:
                return vec(1, 1)
            if self.app.player.grid_pos[0] > COLS//2 and self.app.player.grid_pos[1] < ROWS//2:
                return vec(1, ROWS-2)
            if self.app.player.grid_pos[0] < COLS//2 and self.app.player.grid_pos[1] > ROWS//2:
                return vec(COLS-2, 1)
            else:
                return vec(COLS-2, ROWS-2)

    def time_to_move(self):
        # If Positioned Exactly Within a Collumn
        if int(self.pix_pos.x+TOP_BOTTOM_BUFFER//2) % self.app.cell_width == 0:
            # If Moving Right, Left or Nowhere
            if self.direction == vec(1, 0) or self.direction == vec(-1, 0) or self.direction == vec(0, 0):
                return True

        # If Positioned Exactly Within a Row
        if int(self.pix_pos.y+TOP_BOTTOM_BUFFER//2) % self.app.cell_height == 0:
            # If Moving Down, Up or Nowhere
            if self.direction == vec(0, 1) or self.direction == vec(0, -1) or self.direction == vec(0, 0):
                return True

        # If None of the Above Conditions Met Return False
        return False

    def move(self):
        # Move Randomly or Pathfind to Target?
        if self.personality == "random":
            self.direction = self.get_random_direction()
        else:
            self.direction = self.get_path_direction(self.target)

    def get_path_direction(self, target):
        next_cell = self.find_next_cell_in_path(target)
        xdir = next_cell[0] - self.grid_pos[0]
        ydir = next_cell[1] - self.grid_pos[1]
        return vec(xdir, ydir)

    def find_next_cell_in_path(self, target):
        path = self.BFS([int(self.grid_pos.x), int(self.grid_pos.y)], [
                        int(target[0]), int(target[1])])
        return path[1]

    # Breadth First Search
    def BFS(self, start, target):
        grid = [[0 for x in range(28)] for x in range(30)]
        for cell in self.app.walls:
            if cell.x < 28 and cell.y < 30:
                grid[int(cell.y)][int(cell.x)] = 1
        queue = [start]
        path = []
        visited = []
        while queue:
            current = queue[0]
            queue.remove(queue[0])
            visited.append(current)
            if current == target:
                break
            else:
                neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]
                for neighbour in neighbours:
                    if neighbour[0]+current[0] >= 0 and neighbour[0] + current[0] < len(grid[0]):
                        if neighbour[1]+current[1] >= 0 and neighbour[1] + current[1] < len(grid):
                            next_cell = [neighbour[0] + current[0], neighbour[1] + current[1]]
                            if next_cell not in visited:
                                if grid[next_cell[1]][next_cell[0]] != 1:
                                    queue.append(next_cell)
                                    path.append({"Current": current, "Next": next_cell})
        shortest = [target]
        while target != start:
            for step in path:
                if step["Next"] == target:
                    target = step["Current"]
                    shortest.insert(0, step["Current"])
        return shortest

    def get_random_direction(self):
        while True:
            number = random.randint(-2, 1)
            if number == -2:
                x_dir, y_dir = 1, 0
            elif number == -1:
                x_dir, y_dir = 0, 1
            elif number == 0:
                x_dir, y_dir = -1, 0
            else:
                x_dir, y_dir = 0, -1
            next_pos = vec(self.grid_pos.x + x_dir, self.grid_pos.y + y_dir)
            if next_pos not in self.app.walls:
                break
        return vec(x_dir, y_dir)

    def get_pix_pos(self):
        return vec((self.grid_pos.x*self.app.cell_width)+TOP_BOTTOM_BUFFER//2+self.app.cell_width//2,
                   (self.grid_pos.y*self.app.cell_height)+TOP_BOTTOM_BUFFER//2 +
                   self.app.cell_height//2)
