from pygame.math import Vector2 as vec

# screen settings
WIDTH, HEIGHT = 610, 670
FPS = 120
TOP_BOTTOM_BUFFER = 50
MAZE_WIDTH, MAZE_HEIGHT = WIDTH-TOP_BOTTOM_BUFFER, HEIGHT-TOP_BOTTOM_BUFFER

ROWS = 30
COLS = 28

# colour settings
BLACK = (0, 0, 0)
GREY = (107, 107, 107)
WHITE = (255, 255, 255)

# Define Path Colours
PATHNODE_COLOUR = (71, 191, 234)
PATHENDNODE_COLOUR = (232, 17, 17)

# Define Character Colours
PLAYER_COLOUR = (234, 220, 41)
BLINKY_RED_COLOUR = (208, 22, 22)
PINKY_PINK_COLOUR = (167, 47, 174)
INKY_BLUE_COLOUR = (71, 191, 234)
CLYDE_ORANGE_COLOUR = (229, 128, 15)

# Define Game Over Text Colour
GAMEOVER_TEXT_COLOUR = (208, 22, 22)

# Define Coin Object
COIN_COLOUR = (246, 190, 128)
COIN_RADIUS = 2

# font settings
START_TEXT_SIZE = 20
START_FONT = 'bahnschrift'

# player settings
# PLAYER_START_POS = vec(2, 2)