import pygame
import sys
import copy
from settings import *
from player_class import *
from enemy_class import *


pygame.init()
vec = pygame.math.Vector2

import time
import timeit


class App:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = 'start'
        self.cell_width = MAZE_WIDTH//COLS
        self.cell_height = MAZE_HEIGHT//ROWS
        self.walls = []
        self.coins = []
        self.enemies = []
        self.e_pos = []
        self.p_pos = None
        self.load()
        self.player = None
        self.PacManPlayerControlled = False

        # Define Debug Mode Toggle
        self.debug_mode = False

        self.time_elapsed = 0.0
        self.time_current = 0.0
        self.time_prior = 0.0

        # Define List of AI Types
        self.PacManAITypes = ['None', 'BFS', 'DFS', 'AStar', 'Dijkstra', 'IDDFS', 'BidirectionalAStar', 'BidirectionalBFS', 'BidirectionalDijkstra', 'AStarEuclidean', 'AStarOctile', 'AStarChebyshev', 'BidirectionalAStarEuclidean', 'BidirectionalAStarOctile', 'BidirectionalAStarChebyshev']


    def run(self):
        while self.running:
            if self.state == 'start': # If on Start Screen
                self.start_events()
                self.start_update()
                self.start_draw()
            elif self.state == 'playing': # If on Game Screen
                self.time_current = timeit.default_timer()
                
                self.playing_events()
                self.playing_update()
                self.playing_draw()
                self.clock.tick(FPS)

                if (self.time_current != 0.0):
                    self.time_prior = timeit.default_timer()

                if len(self.coins) != 0:
                    time_delta = self.time_prior - self.time_current
                    self.time_elapsed += time_delta
            elif self.state == 'game over': # If on Game Over Screen
                self.game_over_events()
                self.game_over_update()
                self.game_over_draw()
            else:
                self.running = False
            
        pygame.quit()
        sys.exit()


############################ HELPER FUNCTIONS ##################################
    def draw_text(self, words, screen, pos, size, colour, font_name, centered=False):
        font = pygame.font.SysFont(font_name, size)
        text = font.render(words, False, colour)
        text_size = text.get_size()
        if centered:
            pos[0] = pos[0]-text_size[0]//2
            pos[1] = pos[1]-text_size[1]//2
        screen.blit(text, pos)


    def load(self):
        # Load Maze Background
        self.background = pygame.image.load('maze.png')
        self.background = pygame.transform.scale(self.background, (MAZE_WIDTH, MAZE_HEIGHT))

        # Opening walls file
        # Creating walls list with co-ords of walls
        # stored as  a vector
        with open("walls.txt", 'r') as file:
            for yidx, line in enumerate(file):
                for xidx, char in enumerate(line):
                    if char == "1":
                        self.walls.append(vec(xidx, yidx))
                    elif char == "C":
                        self.coins.append(vec(xidx, yidx))
                    elif char in ["2", "3", "4", "5"]:
                        self.e_pos.append([xidx, yidx])
                    elif char == "P":
                        self.p_pos = [xidx, yidx]
                    elif char == "B":
                        pygame.draw.rect(self.background, BLACK, (xidx*self.cell_width, yidx*self.cell_height,
                                                                  self.cell_width, self.cell_height))


    def make_enemies(self):
        for idx, pos in enumerate(self.e_pos):
            self.enemies.append(Enemy(self, vec(pos), self.screen, idx))


    def draw_grid(self):
        for x in range(WIDTH//self.cell_width):
            pygame.draw.line(self.background, GREY, (x*self.cell_width, 0),
                             (x*self.cell_width, HEIGHT))
        for x in range(HEIGHT//self.cell_height):
            pygame.draw.line(self.background, GREY, (0, x*self.cell_height),
                             (WIDTH, x*self.cell_height))

    def reset(self):
        self.player.lives = 3
        self.player.current_score = 0
        self.player.grid_pos = vec(self.player.starting_pos)
        self.player.pix_pos = self.player.get_pix_pos()
        self.player.direction *= 0
        for enemy in self.enemies:
            enemy.grid_pos = vec(enemy.starting_pos)
            enemy.pix_pos = enemy.get_pix_pos()
            enemy.direction *= 0

        self.coins = []
        with open("walls.txt", 'r') as file:
            for yidx, line in enumerate(file):
                for xidx, char in enumerate(line):
                    if char == 'C':
                        self.coins.append(vec(xidx, yidx))
        self.state = "playing"


########################### INTRO FUNCTIONS ####################################
    # Process Events on the Start Screen !Adam and Arpit Add a Keypress for Your Search Here!
    def start_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            # If Number Key 1 Pressed run Regular PacMan Game
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    self.InitPacMan(0)

            # If Number Key 2 Pressed run PacMan Using BFS
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_2:
                    self.InitPacMan(1)

            # If Number Key 3 Pressed run PacMan Using DFS
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_3:
                    self.InitPacMan(2)

            # If Number Key 4 Pressed run PacMan Using AStar Manhattan
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_4:
                    self.InitPacMan(3)

            # If Number Key 5 Pressed Test PacMan Using Dijkstra
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_5:
                    self.InitPacMan(4)

            # If Number Key 5 Pressed Test PacMan Using IDDFS
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_6:
                    self.InitPacMan(5)

            # If Number Key 5 Pressed Test PacMan Using Bidirectional A*
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_7:
                    self.InitPacMan(6)

            # If Number Key 5 Pressed Test PacMan Using Bidirectional BFS
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_8:
                    self.InitPacMan(7)

            # If Number Key 5 Pressed Test PacMan Using Bidirectional Dijkstra
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_9:
                    self.InitPacMan(8)

            # If Number Key 5 Pressed Test PacMan Using A* Euclidean
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    self.InitPacMan(9)

            # If Number Key 5 Pressed Test PacMan Using A* Octile
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F2:
                    self.InitPacMan(10)

            # If Number Key 5 Pressed Test PacMan Using A* Chebyshev
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F3:
                    self.InitPacMan(11)

            # If Number Key 5 Pressed Test PacMan Using Bidirectional A* Euclidean
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F4:
                    self.InitPacMan(12)

            # If Number Key 5 Pressed Test PacMan Using Bidirectional A* Octile
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F5:
                    self.InitPacMan(13)

            # If Number Key 5 Pressed Test PacMan Using Bidirectional A* Chebyshev
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F6:
                    self.InitPacMan(14)

            # If Number Key 5 Pressed Test PacMan With All Ais
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.InitPacMan('TestAll')


    def start_update(self):
        pass


    # Draw Start Screen !Adam and Arpit Add a List Element For Your Search Here!
    def start_draw(self):
        # Draw Black Background
        self.screen.fill(BLACK)

        # Set List Position
        ListPos = vec(WIDTH//2, HEIGHT//2)

        # Output List of PacMan AIs
        self.draw_text('Pac-Man Search Algorithms:', self.screen, [ListPos.x, ListPos.y - 240], START_TEXT_SIZE, (192, 192, 192), START_FONT, centered=True)
        self.draw_text('[1] Normal Game', self.screen, [ListPos.x, ListPos.y - 210], START_TEXT_SIZE, (197, 36, 36), START_FONT, centered=True)
        self.draw_text('[2] Breadth-First Search', self.screen, [ListPos.x, ListPos.y - 180], START_TEXT_SIZE, (255, 0, 255), START_FONT, centered=True)
        self.draw_text('[3] Depth-First Search', self.screen, [ListPos.x, ListPos.y - 150], START_TEXT_SIZE, (124, 252, 0), START_FONT, centered=True)
        self.draw_text('[4] A* Search (Manhattan distance)', self.screen, [ListPos.x, ListPos.y - 120], START_TEXT_SIZE, (255, 215, 0), START_FONT, centered=True)
        self.draw_text('[5] Dijkstra', self.screen, [ListPos.x, ListPos.y - 90], START_TEXT_SIZE, (148, 0, 211), START_FONT, centered=True)
        self.draw_text('[6] IDDFS', self.screen, [ListPos.x, ListPos.y - 60], START_TEXT_SIZE, (0, 255, 255), START_FONT, centered=True)
        self.draw_text('[7] Bidirectional A* (Manhattan distance)', self.screen, [ListPos.x, ListPos.y - 30], START_TEXT_SIZE, (0, 128, 255), START_FONT, centered=True)
        self.draw_text('[8] Bidirectional BFS', self.screen, [ListPos.x, ListPos.y], START_TEXT_SIZE, (255, 153, 153), START_FONT, centered=True)
        self.draw_text('[9] Bidirectional Dijkstra', self.screen, [ListPos.x, ListPos.y + 30], START_TEXT_SIZE, (255, 128, 0), START_FONT, centered=True)
        self.draw_text('[F1] A* Search (Euclidean distance)', self.screen, [ListPos.x, ListPos.y + 60], START_TEXT_SIZE, (255, 215, 0), START_FONT, centered=True)
        self.draw_text('[F2] A* Search (Octile distance)', self.screen, [ListPos.x, ListPos.y + 90], START_TEXT_SIZE, (148, 0, 211), START_FONT, centered=True)
        self.draw_text('[F3] A* Search (Chebyshev distance)', self.screen, [ListPos.x, ListPos.y + 120], START_TEXT_SIZE, (0, 255, 255), START_FONT, centered=True)
        self.draw_text('[F4] Bidirectional A* (Euclidean distance)', self.screen, [ListPos.x, ListPos.y + 150], START_TEXT_SIZE, (0, 128, 255), START_FONT, centered=True)
        self.draw_text('[F5] Bidirectional A* (Octile distance)', self.screen, [ListPos.x, ListPos.y + 180], START_TEXT_SIZE, (255, 153, 153), START_FONT, centered=True)
        self.draw_text('[F6] Bidirectional A* (Chebyshev distance)', self.screen, [ListPos.x, ListPos.y + 210], START_TEXT_SIZE, (255, 128, 0), START_FONT, centered=True)
        self.draw_text('[T] Test All Pac-Man Search Algorithms', self.screen, [ListPos.x, ListPos.y + 240], START_TEXT_SIZE, (255, 250, 250), START_FONT, centered=True)
        
        # Output Display
        pygame.display.update()


    # Init PacMan Functions !Adam and Arpit Add a Function to Initialize Pacman With Your Search Here!
    # Init Regular PacMan Game
    def InitPacMan(self, AI):
        # Open Game Screen on Next Frame
        self.state = 'playing'

        if type(AI) == int:
            # Create the Player
            self.player = Player(self, vec(self.p_pos), self.screen, self.PacManAITypes[AI])

            # Reg Game or AI
            if (AI == 0): # If Reg Game
                self.player.speed = 2 # Default the players speed

                self.PacManPlayerControlled = True # Player Controlled PacMan
                self.make_enemies() # Populate Enemies

        elif type(AI) == str:
             # Create the Player
            self.player = Player(self, vec(self.p_pos), self.screen, AI)

            # Set PacMan NOT Player Controlled
            self.PacManPlayerControlled = False


########################### PLAYING FUNCTIONS ##################################
    # !FIX PLAYER MOVEMENT HERE LATER!
    def playing_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
                self.debug_mode = not self.debug_mode
                print("Debug Mode: {0}".format(self.debug_mode))
            if self.PacManPlayerControlled and event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.player.move(vec(-1, 0))
                if event.key == pygame.K_RIGHT:
                    self.player.move(vec(1, 0))
                if event.key == pygame.K_UP:
                    self.player.move(vec(0, -1))
                if event.key == pygame.K_DOWN:
                    self.player.move(vec(0, 1))


    def playing_update(self):
        self.player.update()
        for enemy in self.enemies:
            enemy.update()

        for enemy in self.enemies:
            if enemy.grid_pos == self.player.grid_pos:
                self.remove_life()


    def playing_draw(self):
        self.screen.fill(BLACK)
        self.screen.blit(self.background, (TOP_BOTTOM_BUFFER//2, TOP_BOTTOM_BUFFER//2))
        self.draw_coins()
        # self.draw_grid()
        if self.PacManPlayerControlled == True:
            self.draw_text('CURRENT SCORE: {}'.format(self.player.current_score), self.screen, [60, 0], 18, WHITE, START_FONT)
            self.draw_text('HIGH SCORE: 0', self.screen, [WIDTH//2+60, 0], 18, WHITE, START_FONT)
        else:
            self.draw_text('TIME ELAPSED: {0:.2f}'.format(self.time_elapsed), self.screen, [100, 5], 15, WHITE, START_FONT)
            self.draw_text('NODES TRAVERSED: {0}'.format(self.player.NodesTraversed), self.screen, [WIDTH//2+65, 5], 15, WHITE, START_FONT)
        self.player.draw()
        for enemy in self.enemies:
            enemy.draw()
        pygame.display.update()


    def remove_life(self):
        self.player.lives -= 1
        if self.player.lives == 0:
            self.state = "game over"
        else:
            self.player.grid_pos = vec(self.player.starting_pos)
            self.player.pix_pos = self.player.get_pix_pos()
            self.player.direction *= 0
            for enemy in self.enemies:
                enemy.grid_pos = vec(enemy.starting_pos)
                enemy.pix_pos = enemy.get_pix_pos()
                enemy.direction *= 0


    def draw_coins(self):
        for coin in self.coins:
            pygame.draw.circle(self.screen, COIN_COLOUR,
                               (int(coin.x*self.cell_width)+self.cell_width//2+TOP_BOTTOM_BUFFER//2,
                                int(coin.y*self.cell_height)+self.cell_height//2+TOP_BOTTOM_BUFFER//2), COIN_RADIUS)


########################### GAME OVER FUNCTIONS ################################
    def game_over_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.reset()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False


    def game_over_update(self):
        pass


    def game_over_draw(self):
        self.screen.fill(BLACK)
        quit_text = "Press the escape button to QUIT"
        again_text = "Press SPACE bar to PLAY AGAIN"
        self.draw_text("GAME OVER", self.screen, [WIDTH//2, 100],  52, GAMEOVER_TEXT_COLOUR, "arial", centered=True)
        self.draw_text(again_text, self.screen, [
                       WIDTH//2, HEIGHT//2],  36, (190, 190, 190), "arial", centered=True)
        self.draw_text(quit_text, self.screen, [
                       WIDTH//2, HEIGHT//1.5],  36, (190, 190, 190), "arial", centered=True)
        pygame.display.update()
