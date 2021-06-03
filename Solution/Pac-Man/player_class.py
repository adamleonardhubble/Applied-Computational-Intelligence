from collections import OrderedDict
from pathfinding.finder.a_star import AStarFinder
from pathfinding.core.grid import Grid
from pathfinding.core.diagonal_movement import DiagonalMovement
import pygame
from settings import *
import math
vec = pygame.math.Vector2

# Node class
# Used by: A* search


class Node:
    # Node class constructor method (initialisation)
    def __init__(self, position: (), parent: ()):
        self.position = position  # Position of the current node (grid indices)
        # Parent node to the current node (node prior to reaching the current node)
        self.parent = parent
        # Geographical cost (G) of the node - distance from the current node to the start node
        self.g = 0
        # Heuristic cost (H) of the node - distance from the current node to the goal node
        self.h = 0
        # Final cost (F) of the node - accumulation of the geographical cost (G) and heuristic cost (H)
        self.f = 0

    # Class dunder methods (double underscore... method... double underscore...)
    # See: https://www.python-course.eu/python3_magic_methods.php

    # Node comparison method (dunder method)
    # Invocated by all node-wise comparative statements
    def __eq__(self, comparing_node):
        # Return the state of equivalence (Boolean)
        return self.position == comparing_node.position

    # Node sorting method (dunder method)
    # Invocated by .sort() as a key to arrange nodes by their final costs (F) - ascending order implied
    def __lt__(self, comparing_node):
        # Return the state of difference (Boolean)
        return self.f < comparing_node.f


class Player:
    # Init PacMan/Player Function
    def __init__(self, app, pos, input_screen, InputAIType):
        # Init Self
        self.app = app
        #self.starting_pos = [pos.x, pos.y]
        #self.grid_pos = pos
        #self.pix_pos = self.get_pix_pos(self.grid_pos[0], self.grid_pos[1])
        self.direction = vec(1, 0)
        self.stored_direction = None
        self.able_to_move = True
        self.current_score = 0
        self.speed = 4
        self.lives = 1
        self.display_screen = input_screen

        # Init Path Stuff For AI
        self.start = None
        self.target = None
        self.library_implementation_active = False
        self.path = []
        self.heuristic_active = ['manhattan',
                                 'euclidean', 'octile', 'chebyshev']

        # Nodes Traversed Counter
        self.NodesTraversed = 0

        # Test It Counter
        self.TestItCounter = 1

        # List of Coin Vectors (Used to Recreate Coins When Game Reset)
        self.OriginalCoinsArray = []
        for i in range(len(self.app.coins)):
            self.OriginalCoinsArray.append(self.app.coins[i])

        # List of Start Pos (Start on All Coin Positions)
        self.StartPosArray = []
        for i in range(len(self.app.coins)):
            self.StartPosArray.append(self.app.coins[i])

        # Use First Start Pos
        self.starting_pos = [self.StartPosArray[0].x, self.StartPosArray[0].y]
        self.grid_pos = [self.StartPosArray[0].x, self.StartPosArray[0].y]
        self.pix_pos = self.get_pix_pos(self.grid_pos[0], self.grid_pos[1])

        # Init Reg or AI Start Pos and Coins
        if(InputAIType == 'None'):
            self.starting_pos = [pos.x, pos.y]
            self.grid_pos = pos
            self.pix_pos = self.get_pix_pos(self.grid_pos[0], self.grid_pos[1])
        else:
            # Append Reg Pacman Player Pose to Coins and Start Pos Arrays
            self.OriginalCoinsArray.append(pos)
            self.StartPosArray.append(pos)
            self.app.coins.append(pos)

        # Define Step Through All Testing
        self.bTestAll = False
        self.TestAllAICounter = 1
        self.AIType = ""
        if (InputAIType == 'TestAll'):
            # Indicated Testing All AI
            self.bTestAll = True
            # Use First Defined AI
            self.AIType = self.app.PacManAITypes[self.TestAllAICounter]
            # Output
            print("\nTesting All PacMan AIs\nCommencing Testing on {0} PacMan:".format(
                self.AIType))

        else:
            # Assign Relevant AI
            self.AIType = InputAIType

        pygame.display.set_caption(self.AIType)

        # Define Pacman Spritesheet
        self.SpriteSheetImageCounter = 0
        self.PacManSpriteSheet = [pygame.image.load('PacMan_Mouth_Closed.png'), pygame.image.load(
            'PacMan_Mouth_HalfOpen.png'), pygame.image.load('PacMan_Mouth_Open.png'), pygame.image.load('PacMan_Mouth_HalfOpen.png')]

        # Define Output File
        self.OutputFile = "Results/" + self.AIType + "_Results.txt"

        # Create the Output File/Overwrite the Existing Output File
        if(self.AIType != 'None'):
            f = open(self.OutputFile, "w")  # Create File
            # Write Headers
            f.write("Test, Start_Pos, Nodes_Traversed, Time Elapsed")
            f.close()


    # Update Function, Called Once Per Frame !Adam and Arpit Ensure Your Search's Update is Being Called Here!
    def update(self):
        # Execute Relevant PacMan AI Update Function
        if (self.AIType == 'None'):  # If No AI
            self.RegPacManUpd()
        else:
            self.SearchPacManUpd()


    # Different PacMan AI's Update Functions !Adam and Arpit Define an Update Function that Occurs Every Frame for Your Search Here!
    def RegPacManUpd(self):
        # Move if Able
        if self.able_to_move:
            self.pix_pos += self.direction * self.speed

        # Apply a Stored Movement Key Press if Applicable
        if self.time_to_move():  # If PacMan Has Finished Moving to the Next Node in His Path
            # Apply Stored Direction?
            if self.stored_direction != None:  # If a Direction Has Been Stored
                # Set Current Direction to Stored Direction
                self.direction = self.stored_direction

            # Check if able to Move
            self.able_to_move = self.can_move()

        # Get Grid Position From Current Coordinates
        self.grid_pos[0] = (self.pix_pos[0]-TOP_BOTTOM_BUFFER +
                            self.app.cell_width//2)//self.app.cell_width+1
        self.grid_pos[1] = (self.pix_pos[1]-TOP_BOTTOM_BUFFER +
                            self.app.cell_height//2)//self.app.cell_height+1

        # If on a Coin, Eat it
        if self.grid_pos in self.app.coins:
            self.eat_coin()


    def SearchPacManUpd(self):
        # Move to Target
        if self.target != self.grid_pos:  # If Not At Current Target
            # Move
            self.pix_pos += self.direction * self.speed

            # If Finished Moving to Next Node in Path
            if self.time_to_move():
                # Remove Now Current Node From Path
                self.path.remove(self.path[0])
                if(len(self.path) == 0):
                    print("ERROR! Path Empty ERROR!")

                # Set Player Direction to Move Into Next Cell in Path
                #self.direction = vec(self.path[0][0] - self.grid_pos[0], self.path[0][1] - self.grid_pos[1])
                self.SetDir(self.path[0][0] - self.grid_pos[0],
                            self.path[0][1] - self.grid_pos[1])

                # Increment Nodes Traversed Counter
                self.NodesTraversed += 1

        # Get Grid Position From Current Coordinates
        self.grid_pos[0] = (self.pix_pos[0]-TOP_BOTTOM_BUFFER +
                            self.app.cell_width//2)//self.app.cell_width+1
        self.grid_pos[1] = (self.pix_pos[1]-TOP_BOTTOM_BUFFER +
                            self.app.cell_height//2)//self.app.cell_height+1

        # If on a Coin, Eat it
        if self.grid_pos in self.app.coins:
            # Eat the Coin
            self.eat_coin()

            # Search For Another Coin?
            if (self.app.coins == []):  # If There Are No Coins Left
                # Output
                #print("No Coins Left, Nodes Traversed ", self.NodesTraversed)

                # Return Current Grid Pos
                self.target = self.grid_pos

                # Initialize Next Test
                self.InitNextTest()
            else:
                # Get PacMan's Current Position as the Start Pos
                self.start = [int(self.grid_pos[0]), int(self.grid_pos[1])]

                # Use Breadth-First Search to Find the Coin to Pathfind to
                self.target = self.BreadthFirstSearchFindCoin(self.start)

                # Pathfind Using Correct AI Type
                self.PathfindUsingSetAI()


    def PathfindUsingSetAI(self):
        # Set Path Using Set AI Type
        if self.AIType == "BFS":
            self.path = self.BreadthFirstSearch(self.start, self.target)

        elif self.AIType == "DFS":
            self.path = self.DepthFirstSearch(self.start, self.target)

        elif self.AIType == "AStar":
            self.path = self.AStarSearch(
                self.start, self.target, self.library_implementation_active, self.heuristic_active[0])

        elif self.AIType == "AStarEuclidean":
            self.path = self.AStarSearch(
                self.start, self.target, self.library_implementation_active, self.heuristic_active[1])

        elif self.AIType == "AStarOctile":
            self.path = self.AStarSearch(
                self.start, self.target, self.library_implementation_active, self.heuristic_active[2])

        elif self.AIType == "AStarChebyshev":
            self.path = self.AStarSearch(
                self.start, self.target, self.library_implementation_active, self.heuristic_active[3])

        elif self.AIType == "Dijkstra":
            self.path = self.DijkstraSearch(self.start, self.target)

        elif self.AIType == "IDDFS":
            self.path = self.IDDFSearch(self.start, self.target)

        elif self.AIType == "BidirectionalAStar":
            self.path = self.BidirectionalAStarSearch(
                self.start, self.target, self.heuristic_active[0])

        elif self.AIType == "BidirectionalAStarEuclidean":
            self.path = self.BidirectionalAStarSearch(
                self.start, self.target, self.heuristic_active[1])

        elif self.AIType == "BidirectionalAStarOctile":
            self.path = self.BidirectionalAStarSearch(
                self.start, self.target, self.heuristic_active[2])

        elif self.AIType == "BidirectionalAStarChebyshev":
            self.path = self.BidirectionalAStarSearch(
                self.start, self.target, self.heuristic_active[3])

        elif self.AIType == "BidirectionalBFS":
            self.path = self.BidirectionalBreadthFirstSearch(
                self.start, self.target)

        elif self.AIType == "BidirectionalDijkstra":
            self.path = self.BidirectionalDijkstraSearch(
                self.start, self.target)


    # Draw Pac-Man Function
    def draw(self):
        # Rotate PacMan
        rotated_image = self.PacManSpriteSheet[math.floor(
            self.SpriteSheetImageCounter)]
        if(self.direction == vec(-1, 0)):
            rotated_image = pygame.transform.rotate(rotated_image, 180)
        if(self.direction == vec(0, -1)):
            rotated_image = pygame.transform.rotate(rotated_image, 90)
        if(self.direction == vec(0, 1)):
            rotated_image = pygame.transform.rotate(rotated_image, 270)

        # Draw Self
        self.display_screen.blit(
            rotated_image, (self.pix_pos.x - 8, self.pix_pos.y - 8))

        # Increment SpriteSheetImageCounter
        self.SpriteSheetImageCounter = self.SpriteSheetImageCounter + 0.2
        if(self.SpriteSheetImageCounter >= 4):
            self.SpriteSheetImageCounter = 0

        # Drawing player lives
        if (self.AIType == 'None'):  # If in Regular Game
            for x in range(self.lives):  # For Every Life in Array
                pygame.draw.circle(self.app.screen, PLAYER_COLOUR,
                                   (30 + 20*x, HEIGHT - 15), 7)  # Draw It

        # Draw Debug Code
        if self.app.debug_mode == True:
            for x in range(28):
                for y in range(30):
                    if x < 28 and y < 30:
                        cell = self.get_pix_pos(x, y)  # Get Pos on Screen
                        if (x, y) in self.app.coins:
                            pygame.draw.rect(
                                self.app.screen, (187, 0, 255), (cell[0] - 8, cell[1] - 8, 16, 16), True)  # Draw It
                        else:
                            pygame.draw.rect(
                                self.app.screen, (75, 0, 130), (cell[0] - 8, cell[1] - 8, 16, 16), True)  # Draw It

        # Draw Player Path
        if not (self.path == []):  # If There is a Path
            NodePos = [0, 0]
            for PathNode in self.path:  # For Every Node in the Path
                NodePos = self.get_pix_pos(
                    PathNode[0], PathNode[1])  # Get Pos on Screen
                pygame.draw.rect(self.app.screen, PATHNODE_COLOUR,
                                 (NodePos[0] - 8, NodePos[1] - 8, 16, 16), True)  # Draw It
            pygame.draw.rect(self.app.screen, PATHENDNODE_COLOUR,
                             (NodePos[0] - 8, NodePos[1] - 8, 16, 16), True)  # Draw Final Path Node


    # Search algorithms (Shortest Path Problem - SPP)
    #
    # Heuristic Calculation
    def Heuristic(self, start_node, goal_node, using_neighbour=True, heursitic_active=None, neighbouring_node=None):
        # If the heuristic estimate is being calculated for a neighbouring node, do the following
        if using_neighbour == True:
            # If the heuristic specified it Euclidean distance, do the following
            if heursitic_active == "euclidean":
                # Generate heuristics (Euclidean distance - linear distance heuristic)
                # Euclidean distance: https://www.analyticsvidhya.com/blog/2020/02/4-types-of-distance-metrics-in-machine-learning/
                # Calculate the geographical score (G) - distance betwween the current node and the start node
                neighbouring_node.g = math.pow(neighbouring_node.position[0] - start_node[0], 2) + math.pow(
                    neighbouring_node.position[1] - start_node[1], 2)
                # Calculate the heuristic score (H) - distance between the current node and the goal (target) node
                neighbouring_node.h = math.pow(neighbouring_node.position[0] - goal_node[0], 2) + math.pow(
                    neighbouring_node.position[1] - goal_node[1], 2)
                # Calculate the final cost (F) - total cost of the node, the accumulation of the geographical cost (G) and heuristic cost (H)
                neighbouring_node.f = math.sqrt(
                    neighbouring_node.g + neighbouring_node.h)

            # Else if the heuristic specified is Octile distance, do the following
            elif heursitic_active == "octile":
                # Generate heuristics (Octile distance - diagonal-based heuristic)
                # Octile distance: http://theory.stanford.edu/~amitp/GameProgramming/Heuristics.html
                # Calculate the geographical score (G) - distance betwween the current node and the start node
                neighbouring_node.g = abs(neighbouring_node.position[0] - start_node[0]) + abs(
                    neighbouring_node.position[1] - start_node[1])
                # Calculate the heuristic score (H) - distance between the current node and the goal (target) node
                neighbouring_node.h = abs(neighbouring_node.position[0] - goal_node[0]) + abs(
                    neighbouring_node.position[1] - goal_node[1])
                # Calculate the final cost (F) - total cost of the node, the accumulation of the geographical cost (G) and heuristic cost (H)
                neighbouring_node.f = (neighbouring_node.g + neighbouring_node.h) + (
                    math.sqrt(2) - 2) * min(neighbouring_node.g, neighbouring_node.h)

            # Else if the heuristic specified is Chebyshev distance, do the following
            elif heursitic_active == "chebyshev":
                # Generate heuristics (Chebyshev distance - diagonal-based heuristic)
                # Chebyshev distance: https://iq.opengenus.org/chebyshev-distance/
                # Calculate the geographical score (G) - distance betwween the current node and the start node
                neighbouring_node.g = abs(neighbouring_node.position[0] - start_node[0]) + abs(
                    neighbouring_node.position[1] - start_node[1])
                # Calculate the heuristic score (H) - distance between the current node and the goal (target) node
                neighbouring_node.h = abs(neighbouring_node.position[0] - goal_node[0]) + abs(
                    neighbouring_node.position[1] - goal_node[1])
                # Calculate the final cost (F) - total cost of the node, the accumulation of the geographical cost (G) and heuristic cost (H)
                neighbouring_node.f = max(
                    neighbouring_node.g, neighbouring_node.h)

            # Else if no heuristic has been specified (Manhattan distance), do the following
            else:
                # Generate heuristics (Manhattan distance (city block distance) - grid-based heuristic)
                # Manhattan distance: https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_506
                # Calculate the geographical score (G) - distance betwween the current node and the start node
                neighbouring_node.g = abs(neighbouring_node.position[0] - start_node[0]) + abs(
                    neighbouring_node.position[1] - start_node[1])
                # Calculate the heuristic score (H) - distance between the current node and the goal (target) node
                neighbouring_node.h = abs(neighbouring_node.position[0] - goal_node[0]) + abs(
                    neighbouring_node.position[1] - goal_node[1])
                # Calculate the final cost (F) - total cost of the node, the accumulation of the geographical cost (G) and heuristic cost (H)
                neighbouring_node.f = neighbouring_node.g + neighbouring_node.h

        # Else if the heuristic estimate is not being calculated for a neighbouring node, do the following
        else:
            # Generate heuristics (Manhattan distance (city block distance) - grid-based heuristic)
            # Manhattan distance: https://link.springer.com/referenceworkentry/10.1007%2F978-0-387-30164-8_506
            # Calculate the heuristic score (H) - distance betwween the start node and the goal node
            h_score = abs(goal_node[0] - start_node[0]) + \
                abs(goal_node[1] - start_node[1])

            # Return the heuristic cost (H)
            return h_score


    # Find the intersecting node for the the bidirectional variants
    def find_intersection(self, forward_visited, backward_visited):
        # a variable for the the intersection nodes
        intersection = []
        is_it_intersecting = False

        # if forward visited nodes and backward visited nodes have any node/nodes in common
        if len(forward_visited) > 0 and len(backward_visited) > 0:
            # Find the interseccion between the two lists
            set1 = set(tuple(x) for x in forward_visited)
            set2 = set(tuple(x) for x in backward_visited)
            intersection = list(set1 & set2)
            # if intersecting node found
            if len(intersection) > 0:
                is_it_intersecting = True

        # return the intersecting nodes
        return is_it_intersecting, intersection


    # Breadth-First Search (Search only - used by all pathfinding algorithms to locate the coin to pathfind to, so that each has a valid target to traverse towards)
    def BreadthFirstSearchFindCoin(self, start):
        # Init Search Variables
        # Define the Grid to be Searched
        grid = [[0 for x in range(28)] for x in range(30)]
        for cell in self.app.walls:  # Add the Game Walls to the Grid
            if cell.x < 28 and cell.y < 30:
                grid[int(cell.y)][int(cell.x)] = 1
        queue = [start]  # Define a Queue Data Structure of Nodes to Search
        path = []  # Define the Path Data Structure to Remember the BFS Path Through the Grid
        visited = []  # Define the Visited Data Structure of Nodes That Have Been Searched Already

        # Define current node, used in search and later returned
        current = queue[0]

        # BFS Search
        while queue:  # While There are Nodes in the Queue
            current = queue[0]  # Get the First Node in the Queue as Current
            # Remove the Now Current Node From the Queue
            queue.remove(queue[0])
            # Add Current Node to List of Searched Nodes
            visited.append(current)
            if (current in self.app.coins):  # If Current Node is a Coin Break Loop
                break  # Break out of Loop
            else:  # If Current Node is Not a Coin Node, Add All Neighbours to Search Queue
                # Define an Array of Current's Neighbours (Offsets)
                neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]
                for neighbour in neighbours:  # For Every Neighbour
                    # Get the Neighbour Cell Grid Coordinates
                    next_cell = [neighbour[0] + current[0],
                                 neighbour[1] + current[1]]
                    # If Neighbour Within Grid
                    if next_cell[0] >= 0 and next_cell[0] < len(grid[0]) and next_cell[1] >= 0 and next_cell[1] < len(grid) and next_cell not in visited:
                        # If the Neighbour Cell is Not a Wall
                        if grid[next_cell[1]][next_cell[0]] != 1:
                            # Add it to the Queue of Nodes to be Searched
                            queue.append(next_cell)

        # Return the Discovered Coin Node
        return current


    # Breadth First Search
    def BreadthFirstSearch(self, start, goal):
        # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
        distance = self.Heuristic(start, goal, False)

        # If the Manhattan distance between the start and goal nodes is equal to (adjacent) '1' grid cell (do not generate a path), do the following
        if distance == 1:
            # Return the path as the start and goal nodes only
            return [start, goal]

        # Else if the Manhattan distance between the start and target nodes is larger than '1' grid cell (generate a path), do the following
        else:
            #print("Pathing From {0} to {1}".format(start, target))
            # Init Search Variables
            # Define the Grid to be Searched
            grid = [[0 for x in range(28)] for x in range(30)]
            for cell in self.app.walls:  # Add the Game Walls to the Grid
                if cell.x < 28 and cell.y < 30:
                    grid[int(cell.y)][int(cell.x)] = 1
            queue = [start]  # Define a Queue Data Structure of Nodes to Search
            path = []  # Define the Path Data Structure to Remember the BFS Path Through the Grid
            visited = []  # Define the Visited Data Structure of Nodes That Have Been Searched Already

            # BFS Search
            while queue:  # While There are Nodes in the Queue
                # Get the First Node in the Queue as Current
                current = queue[0]
                # Remove the Now Current Node From the Queue
                queue.remove(queue[0])
                # Add Current Node to List of Searched Nodes
                visited.append(current)
                if (goal == current):  # If Current Node is Target Break Out of Loop
                    break  # Break out of Loop
                else:  # If Current Does Not Meet Target Conditions
                    # Define an Array of Current's Neighbours (Offsets)
                    neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]
                    for neighbour in neighbours:  # For Every Neighbour
                        # Get the Neighbour Cell Grid Coordinates
                        next_cell = [neighbour[0] + current[0],
                                     neighbour[1] + current[1]]
                        # If Neighbour Within Grid
                        if next_cell[0] >= 0 and next_cell[0] < len(grid[0]) and next_cell[1] >= 0 and next_cell[1] < len(grid) and next_cell not in visited:
                            # If the Neighbour Cell is Not a Wall
                            if grid[next_cell[1]][next_cell[0]] != 1:
                                # Add it to the Queue of Nodes to be Searched
                                queue.append(next_cell)
                                # Add the Path Taken to the Neighbour Cell to Path
                                path.append(
                                    {"Start": current, "Dest": next_cell})

            # Working Backwards From target Find the Path Taken to Reach It (NOTE: This assumes that target was found)
            shortest = [goal]  # Preset Intial Path to Contain only Target Node
            while goal != start:  # While the Start Pos Hasn't Yet Been Reached
                for step in path:  # Loop Through Every Node With a Path Stored in path
                    if step["Dest"] == goal:  # If the BFS Acessed the Current Target From This Node
                        # Make the New Current Target the One the Old Target was Acessed From
                        goal = step["Start"]
                        # Insert the Old Target at the Start of the Shortest Path
                        shortest.insert(0, goal)

            # Return the Completed Path
            return shortest


    # Depth First Search
    def DepthFirstSearch(self, start, goal):
       # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
        distance = self.Heuristic(start, goal, False)

        # If the Manhattan distance between the start and goal nodes is equal to (adjacent) '1' grid cell (do not generate a path), do the following
        if distance == 1:
            # Return the path as the start and goal nodes only
            return [start, goal]

        # Else if the Manhattan distance between the start and target nodes is larger than '1' grid cell (generate a path), do the following
        else:
            # Init Search Variables
            # Define the Grid to be Searched
            grid = [[0 for x in range(28)] for x in range(30)]
            for cell in self.app.walls:  # Add the Game Walls to the Grid
                if cell.x < 28 and cell.y < 30:
                    grid[int(cell.y)][int(cell.x)] = 1
            stack = [start]  # Define a Stack Data Structure of Nodes to Search
            path = []  # Define the Path Data Structure to Remember the BFS Path Through the Grid
            visited = []  # Define the Visited Data Structure of Nodes That Have Been Searched Already

            # Depth-First Search
            while stack:  # While There are Nodes Still in the Stack
                # 'Pop Off' the Node on the top of the Stack as the Current Node
                current = stack[0]
                # Remove the Now Current Node From the Stack
                stack.remove(stack[0])
                # Add Current Node to List of Searched Nodes
                visited.append(current)
                if (goal == current):  # If Current Meets Target Conditions Break Out of Loop
                    break  # Break out of Loop
                else:  # If Current Does Not Meet Target Conditions
                    # Define an Array of Current's Neighbours (Offsets)
                    neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]
                    for neighbour in neighbours:  # For Every Neighbouring Cell
                        # Get the Neighbouring Cells Grid Coordinates
                        next_cell = [neighbour[0] + current[0],
                                     neighbour[1] + current[1]]
                        # If Neighbour Coordinates Within Grid and Hasn't Been Visited Yet
                        if next_cell[0] >= 0 and next_cell[0] < len(grid[0]) and next_cell[1] >= 0 and next_cell[1] < len(grid) and next_cell not in visited:
                            # If the Neighbour Cell is Not a Wall
                            if grid[next_cell[1]][next_cell[0]] != 1:
                                # Add it to the Stack of Nodes to be Searched
                                stack.insert(0, next_cell)
                                # Add the Path Taken to the Neighbour Cell to Path
                                path.append(
                                    {"Start": current, "Dest": next_cell})

            # Working Backwards From target Find the Path Taken to Reach It (NOTE: This assumes that target was found)
            shortest = [goal]  # Preset Intial Path to Contain only Target Node
            while goal != start:  # While the Start Pos Has Not Yet Been Reached
                for step in path:  # Loop Through Every Node With a Path Stored in path
                    if step["Dest"] == goal:  # If the DFS Acessed the Current Target From This Node
                        # Make the New Current Target the One the Old Target was Acessed From
                        goal = step["Start"]
                        # Insert the Old Target at the Start of the Shortest Path
                        shortest.insert(0, goal)

            # Return the Completed Path
            return shortest


    # Dijkstra Search
    def DijkstraSearch(self, start, goal):
        # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
        distance = self.Heuristic(start, goal, False)

        # If the Manhattan distance between the start and goal nodes is equal to (adjacent) '1' grid cell (do not generate a path), do the following
        if distance == 1:
            # Return the path as the start and goal nodes only
            return [start, goal]

        # Else if the Manhattan distance between the start and target nodes is larger than '1' grid cell (generate a path), do the following
        else:
            # Define Grid Array
            # Define the Grid to be Searched
            grid = [[0 for x in range(30)] for x in range(28)]
            for cell in self.app.walls:  # Add the Game Walls to the Grid
                if cell.x < 30 and cell.y < 28:
                    grid[int(cell.x)][int(cell.y)] = 1

            # Define Node G Score Array
            NodeGScore = [[0 for x in range(30)]
                          for x in range(28)]  # Define the Array
            GScoreCheckQueue = [start]
            NodeGScore[start[0]][start[1]] = 0

            GScoreCheckQueueIndex = 0
            while(GScoreCheckQueueIndex < len(GScoreCheckQueue)):  # Loop Until Entire Queue Checked
                # Get Current Node
                CurrentNode = GScoreCheckQueue[GScoreCheckQueueIndex]
                CurrentNodeGScore = NodeGScore[CurrentNode[0]][CurrentNode[1]]

                # Define Neighbours Array (Offsets from Current)
                neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]

                # Add Every Neighbour to Check Queue
                for neighbour in neighbours:
                    # Get the Neighbour Cell Index
                    # Get Neighbour Cell
                    NeighbourCell = [
                        neighbour[0] + CurrentNode[0], neighbour[1] + CurrentNode[1]]
                    # If Within Grid Bounds
                    if(NeighbourCell[0] >= 0 and NeighbourCell[0] < len(grid) and NeighbourCell[1] >= 0 and NeighbourCell[1] < len(grid[0])):
                        if (NeighbourCell not in GScoreCheckQueue):  # If Not Already in Queue
                            if(grid[NeighbourCell[0]][NeighbourCell[1]] != 1):  # If Cell Not a Wall
                                # Append Neighbour Cell to Checked List
                                GScoreCheckQueue.append(NeighbourCell)
                                # Assign Node G Score
                                NodeGScore[NeighbourCell[0]
                                           ][NeighbourCell[1]] = CurrentNodeGScore + 1
                            else:
                                # Assign G Score for Wall
                                NodeGScore[NeighbourCell[0]
                                           ][NeighbourCell[1]] = 10000

                # Increment Counter
                GScoreCheckQueueIndex += 1

            # Define Path to Return
            path = [goal]
            while (goal != start):
                # Append Neighbour With Lowest G Score
                LowestNeighbour = goal
                targetGScore = NodeGScore[goal[0]][goal[1]]

                # Get the Neighbour Cell Index
                neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]
                for neighbour in neighbours:
                    NeighbourCell = [neighbour[0] +
                                     goal[0], neighbour[1] + goal[1]]
                    # If Within Grid Bounds
                    if(NeighbourCell[0] >= 0 and NeighbourCell[0] < len(grid) and NeighbourCell[1] >= 0 and NeighbourCell[1] < len(grid[0])):
                        if (grid[NeighbourCell[0]][NeighbourCell[1]] != 1):  # If Not a Wall
                            if (NodeGScore[NeighbourCell[0]][NeighbourCell[1]] == (targetGScore - 1)):
                                LowestNeighbour = NeighbourCell

                # Append Lowest Neighbour
                path.append(LowestNeighbour)

                # Lowest Neighbour Becomes New Target
                goal = LowestNeighbour

            # Return Path in Reverse
            return list(reversed(path))

    # A* Search
    def AStarSearch(self, start, goal=None, library_active=False, heursitic_active=None):
        # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
        distance = self.Heuristic(start, goal, False)

        # If the Manhattan distance between the start and goal nodes is equal to (adjacent) '1' grid cell (do not generate a path), do the following
        if distance == 1:
            # Return the path as the start and goal nodes only
            return [start, goal]

        # Else if the Manhattan distance between the start and target nodes is larger than '1' grid cell (generate a path), do the following
        else:
            # If the library implementation is being used, do the following
            if library_active == True:
                # Initialise the grid, all elements are set '1' initially (representing walkable space)
                # See here: https://pypi.org/project/pathfinding/
                grid = [[1 for x in range(28)] for x in range(30)]

                # For each cell in the wall array, do the following
                for cell in self.app.walls:
                    # If the iterated cell is located within the grid space, do the following
                    if cell.x < 28 and cell.y < 30:
                        # Update the iterated cells value to '0' (representing wall space)
                        grid[int(cell.y)][int(cell.x)] = 0

                # Compile a grid type object from the map representation given
                grid = Grid(matrix=grid)

                # Set the start node for pac-man
                start_node = grid.node(start[0], start[1])
                # Set the goal (target) node for pac-man
                goal_node = grid.node(goal[0], goal[1])

                # Configure the search properties of A*, create a finder instance of the algorithm
                # Never diagonal movement is allowed, grid only consists of single-run traversable spaces
                finder = AStarFinder(
                    diagonal_movement=DiagonalMovement.only_when_no_obstacle)
                # Compile the optimal path and number of executions required to compute the path
                path, runs = finder.find_path(start_node, goal_node, grid)

                return path  # Return the completed path generated from pac-mans start node, to the goal node passed

            # Else if the library implementation is not being used, do the following
            else:
                # Initialise the grid, all elements are set '0' initially (representing walkable space)
                grid = [[0 for x in range(28)] for x in range(30)]

                # For each cell in the wall array, do the following
                for cell in self.app.walls:
                    # If the iterated cell is located within the grid space, do the following
                    if cell.x < 28 and cell.y < 30:
                        # Update the iterated cells value to '1' (representing wall space)
                        grid[int(cell.y)][int(cell.x)] = 1

                open_list = []  # Create the list of nodes that have to have their costs evaluated/ calculated
                closed_list = []  # Create the list of nodes that have had their costs evaluated/ calculated

                # Create a start node from where pac-man intially traverses from
                start_node = Node(start, None)
                # Create a goal node (target) to where pac-man traverses to
                goal_node = Node(goal, None)

                # Append the start node to the open list (cost has been evaluated upon initialisation)
                open_list.append(start_node)

                # While the list of nodes to be evaluated contains nodes (evaluate all nodes considered for the path), do the following
                while len(open_list) > 0:
                    # cheapest_node = Node(None, None) # Create a node object instance, representing the cheapest node from the current node
                    # cheapest_node_index = 0 # Create an integer representing the index of the cheapest node in the open list

                    # For each node contained in the list of nodes that have not already had their costs evaluated/ calculated, do the following
                    # for i, node in enumerate(open_list):
                    # If the currently iterated nodes cost is cheaper than the cheapest known nodes cost, do the following
                    # if node.f < cheapest_node.f:
                    # cheapest_node.f = node.f # Update the cheapest known nodes cost to the currently iterated nodes cost
                    # cheapest_node_index = i # Update the cheapest known nodes index to the currently iterated nodes index in the open list

                    open_list.sort()  # Sort the list of nodes to be evaluated in ascending order, with the lowest final cost (F) node positioned first

                    # current_node = open_list[cheapest_node_index] # Set the current node being evaluated to the node needing to be evaluated with the lowest final cost (F)
                    # Set the current node being evaluated to the node needing to be evaluated with the lowest final cost (F)
                    current_node = open_list[0]
                    # Remove the current node from the open list of nodes needing to be evaluated
                    open_list.pop(0)

                    # Append the current node to the closed list of nodes already evaluated
                    closed_list.append(current_node)

                    # If the current node being evaluated is the goal node, do the following
                    if current_node.position == goal_node.position:
                        path = []  # Create a path of nodes representing the composition of the route to reach the goal node from the start node

                        # While the current node being evaluated is not the start node (path has not been fully generated yet), do the following
                        while current_node.position != start_node.position:
                            # Append the position (grid indices) of the current node to the generated path
                            path.append(current_node.position)
                            # Set the current node to be its parent (node in the path prior), allowing the path to be reversed populated
                            current_node = current_node.parent

                        # Append the start node to the path generated, as the first node
                        path.append(start)
                        # Inverse the order of nodes comprising the generated path to represent pac-mans path relative to its current node to the goal node
                        path = list(reversed(path))

                        return path  # Return the completed path generated from pac-mans start node, to the goal node passed

                    # Create the array of possible directions from pac-mans position to neighbouring nodes (no diagonal movements - four directions only)
                    neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]

                    # closed_list_node_positions = [] # Create the array of closed list node positions

                    # For each node in the list of nodes that already have their costs evaluated/ calculated, do the following
                    # for node in closed_list:
                    # closed_list_node_positions.append(node.position) # Append the position of the current node to the array of node positions comprising the closed list

                    # For each direction associated to a possible neighbouring node in the array of possible directions, do the following
                    for neighbour in neighbours:
                        # Calculate the corresponding position of the neighbouring node
                        neighbour_node = [
                            neighbour[0] + current_node.position[0], neighbour[1] + current_node.position[1]]

                        # If the neigbouring node formulated is within the boundaries of the grid, do the following
                        if neighbour_node[0] >= 0 and neighbour_node[0] < len(grid[0]) and neighbour_node[1] >= 0 and neighbour_node[1] < len(grid):
                            # Store the value of the grid node at the position of the neigbouring node (is it a wall or a traversable node?)
                            grid_node_value = grid[neighbour_node[1]
                                                   ][neighbour_node[0]]

                            # If the value of the grid node is not equal to '1' (grid node is not a wall), do the following
                            if grid_node_value != 1:
                                # Create a node object instance, representing the neighbouring node
                                neighbouring_node = Node(
                                    neighbour_node, current_node)

                                # If the neighbouring nodes final cost (F) has not yet been evaluated (not in the closed list), do the following
                                # if neighbouring_node.position not in closed_list_node_positions:
                                if neighbouring_node not in closed_list:
                                    # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
                                    self.Heuristic(
                                        start_node.position, goal_node.position, True, heursitic_active, neighbouring_node)

                                    # Create a Boolean state, representing whether the neighbour node is going to be added to the open list and considered in the generated path
                                    neighbour_considered_in_path = True

                                    # For each node in the list of nodes that have not had their costs evaluated/ calculated, do the following
                                    for node in open_list:
                                        # If the neigbouring node shares the same position with the currently iterated node and is equally or more costly than it, do the following
                                        if neighbouring_node.position == node.position and neighbouring_node.f >= node.f:
                                            # Do not consider the node for the path being generated, it is already conatined in the open list and is costly
                                            # The neigbouring node is not being considered in the generated path
                                            neighbour_considered_in_path = False

                                        # Else if the neigbouring node does not share the same position with the currently iterated node and is less costly than it, do the following
                                        else:
                                            # Consider the node for the path being generated, it is not already contained in the open list or it is inexpensive
                                            # The neigbouring node is being considered in the generated path
                                            neighbour_considered_in_path = True

                                    # If the neighbouring node is being considered in the generated path, do the following
                                    if neighbour_considered_in_path == True:
                                        # Append the neighbouring node to the list of nodes that have not had their cost evaluated/ calculated
                                        open_list.append(neighbouring_node)


    # Iterative Deepening Depth-First Search
    def IDDFSearch(self, start, goal):
        # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
        distance = self.Heuristic(start, goal, False)

        # If the Manhattan distance between the start and goal nodes is equal to (adjacent) '1' grid cell (do not generate a path), do the following
        if distance == 1:
            # Return the path as the start and goal nodes only
            return [start, goal]

        # Else if the Manhattan distance between the start and target nodes is larger than '1' grid cell (generate a path), do the following
        else:
            #print("Pathing From {0} to {1}".format(start, goal))
            # Define List of Checked Nodes
            CheckedNodesList = []

            # Define Path List
            Path = []

            # Define Grid Array
            # Define the Grid to be Searched
            grid = [[0 for x in range(30)] for x in range(28)]
            for cell in self.app.walls:  # Add the Game Walls to the Grid
                if cell.x < 30 and cell.y < 28:
                    grid[int(cell.x)][int(cell.y)] = 1

            # Define Depth Limited DFS
            def DLS(node, depth):
                # Search on Basis of Passed Depth
                if (depth == 0):  # If 0 Search Depth (Only Searching Current Node)

                    # Return if Current Node is Target
                    if (node == goal):
                        Path.append(goal)
                        return (node, True)
                    else:
                        return (None, True)
                elif (depth > 0):  # If Positive Search Depth
                    # Define any_remaining
                    any_remaining = False

                    # Define Neighbours Array (Offsets)
                    neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]

                    # Add Every Neighbour to Check Queue
                    for neighbour in neighbours:
                        # Get the Neighbour Cell Index
                        # Get the Neighbour Cell
                        NeighbourCell = [neighbour[0] +
                                         node[0], neighbour[1] + node[1]]
                        # If Neighbour Cell Within Grid Bounds
                        if(NeighbourCell[0] >= 0 and NeighbourCell[0] < len(grid) and NeighbourCell[1] >= 0 and NeighbourCell[1] < len(grid[0])):
                            if (NeighbourCell not in CheckedNodesList):  # If Not Already in Queue
                                # If Cell Not a Wall
                                if(grid[NeighbourCell[0]][NeighbourCell[1]] != 1):
                                    # Append Current Node to CheckedNodesList
                                    CheckedNodesList.append(NeighbourCell)

                                    # Recursively Call DLS on Neighbour Cell With One Less Depth
                                    found, remaining = DLS(
                                        NeighbourCell, depth - 1)

                                    if (found != None):
                                        Path.append(NeighbourCell)
                                        return(NeighbourCell, True)

                                    if remaining:
                                        any_remaining = True

                    # Return
                    return (None, any_remaining)
                else:  # If Negative Search Depth (This is an Error)
                    print("ERROR! Negative Search Depth Passed ERROR!")

            # Define Search Depth
            search_depth = 0
            while (True):
                CheckedNodesList = []
                # Do the Search
                found, remaining = DLS(start, search_depth)

                # Increment Depth
                search_depth = search_depth + 1

                # Return Path
                if (found != None):
                    Path.append(start)
                    Path = Path[::-1]
                    del Path[-1]

                    if(len(Path) >= 3):
                        if(Path[0] == Path[2]):
                            del Path[0]
                            del Path[0]

                    #print("Path: {0}".format(Path))
                    return Path


    # Bidirectional A* Search
    def BidirectionalAStarSearch(self, start, goal=None, heuristic_active=None):
        # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
        distance = self.Heuristic(start, goal, False)

        # If the Manhattan distance between the start and goal nodes is equal to (adjacent) '1' grid cell (do not generate a path), do the following
        if distance == 1:
            # Return the path as the start and goal nodes only
            return [start, goal]

        # Else if the Manhattan distance between the start and target nodes is larger than '1' grid cell (generate a path), do the following
        else:
            # Initialise the grid, all elements are set '0' initially (representing walkable space)
            grid = [[0 for x in range(28)] for x in range(30)]

            # For each cell in the wall array, do the following
            for cell in self.app.walls:
                # If the iterated cell is located within the grid space, do the following
                if cell.x < 28 and cell.y < 30:
                    # Update the iterated cells value to '1' (representing wall space)
                    grid[int(cell.y)][int(cell.x)] = 1

            # Create the list of nodes that have to have their costs evaluated/ calculated
            forward_open_list = []
            # Create the list of nodes that have had their costs evaluated/ calculated
            forward_closed_list = []

            # Create a start node from where pac-man intially traverses from
            forward_start_node = Node(start, None)
            # Create a goal node (target) to where pac-man traverses to
            forward_goal_node = Node(goal, None)

            # Append the start node to the open list (cost has been evaluated upon initialisation)
            forward_open_list.append(forward_start_node)

            # Create the list of nodes that have to have their costs evaluated/ calculated
            backward_open_list = []
            # Create the list of nodes that have had their costs evaluated/ calculated
            backward_closed_list = []

            # Create a start node from where pac-man intially traverses from
            backward_start_node = Node(goal, None)
            # Create a goal node (target) to where pac-man traverses to
            backward_goal_node = Node(start, None)

            # Append the start node to the open list (cost has been evaluated upon initialisation)
            backward_open_list.append(backward_start_node)

            # While the list of nodes to be evaluated contains nodes (evaluate all nodes considered for the path), do the following
            while len(forward_open_list) > 0 and len(backward_open_list) > 0:
                # Sort the list of nodes to be evaluated in ascending order, with the lowest final cost (F) node positioned first
                forward_open_list.sort()

                # current_node = open_list[cheapest_node_index] # Set the current node being evaluated to the node needing to be evaluated with the lowest final cost (F)
                # Set the current node being evaluated to the node needing to be evaluated with the lowest final cost (F)
                forward_current_node = forward_open_list[0]
                # Remove the current node from the open list of nodes needing to be evaluated
                forward_open_list.pop(0)

                # Append the current node to the closed list of nodes already evaluated
                forward_closed_list.append(forward_current_node)

                # Sort the list of nodes to be evaluated in ascending order, with the lowest final cost (F) node positioned first
                backward_open_list.sort()

                # current_node = open_list[cheapest_node_index] # Set the current node being evaluated to the node needing to be evaluated with the lowest final cost (F)
                # Set the current node being evaluated to the node needing to be evaluated with the lowest final cost (F)
                backward_current_node = backward_open_list[0]
                # Remove the current node from the open list of nodes needing to be evaluated
                backward_open_list.pop(0)

                # Append the current node to the closed list of nodes already evaluated
                backward_closed_list.append(backward_current_node)

                # If the current node being evaluated is the goal node, do the following
                if forward_current_node.position == backward_current_node.position or forward_current_node.position == forward_goal_node.position or backward_current_node.position == backward_goal_node.position:
                    # Inverse the order of nodes comprising the generated path to represent pac-mans path relative to its current node to the goal node
                    forward_path = self.astar_path_deduction(
                        forward_current_node, forward_start_node, start)
                    # Inverse the order of nodes comprising the generated path to represent pac-mans path relative to its current node to the goal node
                    backward_path = self.astar_path_deduction(
                        backward_current_node, backward_start_node, goal)

                    immediate_intersecting_node = []  # Create the the intersecting node array
                    # Create the intersecting nodes index in the corresponding generated path array
                    intersecting_node_index = 0

                    # For each node comprising the forward generated path, do the following
                    for i, node in enumerate(forward_path):
                        # If the currently iterated node also comprises the backward generated path, do the following
                        if node in backward_path:
                            # Set the node immediately intersecting the generated paths to the currently iterated node
                            immediate_intersecting_node = node
                            # Store the index of the immediately intersecting node in the generated paths
                            intersecting_node_index = i

                            break  # Break from the iterative statement

                    # Inverse the order of nodes comprising the generated path to represent pac-mans path relative to the intersecting node and the goal node
                    backward_path = list(reversed(backward_path))

                    # For each node comprising the backward generated path, do the following
                    for i, node in enumerate(backward_path):
                        # If the currently iterated node is the node immediately intersecting the generated paths
                        if node == immediate_intersecting_node:
                            break  # Break from the iterative statement

                        # Else if the currently iterated node exists prior to the immediately intersecting node in the generated paths (ignore), do the following
                        else:
                            # Update the value of the node in the backward path to a wildcard value of 'None' (to be ignored in the upcoming path revision procedure)
                            backward_path[i] = None

                    revised_backward_path = []  # Create a revised backward path array

                    # For each node comprising the backward generated path, do the following
                    for node in backward_path:
                        # If the currently iterated node is contained in the generated path intersection, do the following
                        if node != None:
                            # Append the currently iterated node to the revised backward generated path
                            revised_backward_path.append(node)

                    revised_forward_path = []  # Create a revised forward path array

                    # For each node comprising the forward generated path, do the following
                    for i, node in enumerate(forward_path):
                        # If the currently iterated node is contained in the generated path intersection, do the following
                        if i < intersecting_node_index:
                            # Append the currently iterated node to the revised forward generated path
                            revised_forward_path.append(node)

                    # If the goal node does not exist in the revised backward generated path, do the following
                    if start not in revised_forward_path:
                        # Append the goal node to the revised backward generated path
                        revised_forward_path.insert(0, goal)

                    # If the goal node does not exist in the revised backward generated path, do the following
                    if goal not in revised_backward_path:
                        # Append the goal node to the revised backward generated path
                        revised_backward_path.append(goal)

                    # Merge the revised forward and backward generated paths to formulate the complete generated path
                    complete_path = revised_forward_path + revised_backward_path

                    # Return the completed path generated from pac-mans start node, to the goal node passed
                    return complete_path

                # Append the neighbouring node to the list of nodes that have not had their cost evaluated/ calculated
                forward_current_node, forward_closed_list, heursitic_active, forward_start_node, forward_goal_node, forward_open_list = self.astar_path_exploration(
                    forward_current_node, forward_closed_list, heuristic_active, forward_start_node, forward_goal_node, forward_open_list, grid)

                # Append the neighbouring node to the list of nodes that have not had their cost evaluated/ calculated
                backward_current_node, backward_closed_list, heursitic_active, backward_start_node, backward_goal_node, backward_open_list = self.astar_path_exploration(
                    backward_current_node, backward_closed_list, heuristic_active, backward_start_node, backward_goal_node, backward_open_list, grid)


    # The method is used to deduce the shortest path from all the cells explored in the grid using the A* Algorithm
    def astar_path_deduction(self, current_node, start_node, start):
        path = []  # Create a path of nodes representing the composition of the route to reach the goal node from the start node

        # While the current node being evaluated is not the start node (path has not been fully generated yet), do the following
        while current_node.position != start_node.position:
            # Append the position (grid indices) of the current node to the generated path
            path.append(current_node.position)
            # Set the current node to be its parent (node in the path prior), allowing the path to be reversed populated
            current_node = current_node.parent

        # Append the start node to the path generated, as the first node
        path.append(start)

        # Inverse the order of nodes comprising the generated path to represent pac-mans path relative to its current node to the goal node
        path = list(reversed(path))

        # Return the algorithm related parameters
        return path


    # Explore the  cells in the grid starting from 'start' node to 'goal' node
    def astar_path_exploration(self, current_node, closed_list, heuristic_active, start_node, goal_node, open_list, grid):
        # Create the array of possible directions from pac-mans position to neighbouring nodes (no diagonal movements - four directions only)
        neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]

        # For each direction associated to a possible neighbouring node in the array of possible directions, do the following
        for neighbour in neighbours:
            # Calculate the corresponding position of the neighbouring node
            neighbour_node = [neighbour[0] + current_node.position[0],
                              neighbour[1] + current_node.position[1]]

            # If the neigbouring node formulated is within the boundaries of the grid, do the following
            if neighbour_node[0] >= 0 and neighbour_node[0] < len(grid[0]) and neighbour_node[1] >= 0 and neighbour_node[1] < len(grid):
                # Store the value of the grid node at the position of the neigbouring node (is it a wall or a traversable node?)
                grid_node_value = grid[neighbour_node[1]][neighbour_node[0]]

                # If the value of the grid node is not equal to '1' (grid node is not a wall), do the following
                if grid_node_value != 1:
                    # Create a node object instance, representing the neighbouring node
                    neighbouring_node = Node(neighbour_node, current_node)

                    # If the neighbouring nodes final cost (F) has not yet been evaluated (not in the closed list), do the following
                    # if neighbouring_node.position not in closed_list_node_positions:
                    if neighbouring_node not in closed_list:
                        # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
                        self.Heuristic(start_node.position, goal_node.position,
                                       True, heuristic_active, neighbouring_node)

                        # Create a Boolean state, representing whether the neighbour node is going to be added to the open list and considered in the generated path
                        neighbour_considered_in_path = True

                        # For each node in the list of nodes that have not had their costs evaluated/ calculated, do the following
                        for node in open_list:
                            # If the neigbouring node shares the same position with the currently iterated node and is equally or more costly than it, do the following
                            if neighbouring_node.position == node.position and neighbouring_node.f >= node.f:
                                # Do not consider the node for the path being generated, it is already conatined in the open list and is costly
                                # The neigbouring node is not being considered in the generated path
                                neighbour_considered_in_path = False

                            # Else if the neigbouring node does not share the same position with the currently iterated node and is less costly than it, do the following
                            else:
                                # Consider the node for the path being generated, it is not already contained in the open list or it is inexpensive
                                # The neigbouring node is being considered in the generated path
                                neighbour_considered_in_path = True

                        # If the neighbouring node is being considered in the generated path, do the following
                        if neighbour_considered_in_path == True:
                            # Append the neighbouring node to the list of nodes that have not had their cost evaluated/ calculated
                            open_list.append(neighbouring_node)

        # Return the algorithm related parameters
        return current_node, closed_list, heuristic_active, start_node, goal_node, open_list


    # Bidirectional Breadth-First Search
    def BidirectionalBreadthFirstSearch(self, start, goal):
        # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
        distance = self.Heuristic(start, goal, False)

        # If the Manhattan distance between the start and goal nodes is equal to (adjacent) '1' grid cell (do not generate a path), do the following
        if distance == 1:
            # Return the path as the start and goal nodes only
            return [start, goal]

        # Else if the Manhattan distance between the start and target nodes is larger than '1' grid cell (generate a path), do the following
        else:
            # Variables for the shortest distance from both ends
            forward_shortest = []
            backward_shortest = []

            # Define the Grid to be Searched
            grid = [[0 for x in range(28)] for x in range(30)]
            for cell in self.app.walls:  # Add the Game Walls to the Grid
                if cell.x < 28 and cell.y < 30:
                    grid[int(cell.y)][int(cell.x)] = 1

            # Common intersecting node variable
            intersecting_node = []

            # Define a Open Queue Data Structure of Nodes to Search
            forward_open_queue = [start]
            # Define the Graph Tree Data Structure to Remember the BFS Path Through the Grid
            forward_graph_tree = []
            # Define the Visited Nodes / Closed Queue Data Structure of Nodes That Have Been Searched Already
            forward_closed_queue = [start]

            # Define a Queue Data Structure of Nodes to Search
            backward_open_queue = [goal]
            # Define the Path Data Structure to Remember the BFS Path Through the Grid
            backward_graph_tree = []
            # Define the Visited Data Structure of Nodes That Have Been Searched Already
            backward_closed_queue = [goal]

            # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
            distance = self.Heuristic(start, goal, False)
            intersection = []  # Variable for intersecting node/cell between the two explorations

            # BFS Search
            while forward_open_queue and backward_open_queue:  # While There are Nodes in the Queue

                # Explore nodes/cells in forward direction (from start to intersecting cell)
                forward_open_queue,  forward_closed_queue, forward_graph_tree = self.bfs_exploration(
                    forward_open_queue, grid, forward_closed_queue, forward_graph_tree)

                # Explore nodes/cells in backward direction (from target to intersecting cell)
                backward_open_queue,  backward_closed_queue, backward_graph_tree = self.bfs_exploration(
                    backward_open_queue, grid, backward_closed_queue, backward_graph_tree)

                # Check for the intersecting node/cells between 2 BFSs running from opposite ends
                # If there is/are common cells between forward visited cells and backward visited cells,
                # Make those cells intersection cells and break out of the while loop
                is_it_intersection, intersection = self.find_intersection(
                    forward_closed_queue, backward_closed_queue)
                if is_it_intersection == True:
                    break

            # if the list is not empty
            if len(forward_closed_queue) != 0:
                # make intersecting cell/node the target for the forward direction
                forward_target = list(intersection[0])
                # Deduce shortest path from the explored nodes/cells in forward direction till intersecting cell/node
                forward_shortest = self.bfs_path_deduction(
                    start=start, goal=forward_target, graph_tree=forward_graph_tree)

            # if the list is not empty
            if len(backward_closed_queue) != 0:
                # make intersecting cell/node the target for the backward direction
                backward_target = list(intersection[-1])
                # Deduce shortest path from the explored nodes/cells in backward direction till intersecting cell/node
                backward_shortest = self.bfs_path_deduction(
                    start=goal, goal=backward_target,  graph_tree=backward_graph_tree)

            # Reverse the path calculated from
            backward_shortest = list(reversed(backward_shortest))

            # Remove any common cells for paths calculated by BFS from both sides
            for cell in forward_shortest:
                if cell in backward_shortest:
                    backward_shortest.pop(backward_shortest.index(cell))

            # The complete path composed of paths calculated by BFS from both sides
            complete_shortest = forward_shortest + backward_shortest

            # return the complete path
            return complete_shortest


    # Explore the cells in the grid starting from 'start' node to 'goal' node
    def bfs_exploration(self, open_queue, grid, closed_queue, graph_tree):
        # Offsets for calculating adjacent cells
        neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]

        # Get the First Node in the Queue as Current and Remove the Current Node From the Queue
        forward_current = open_queue.pop(0)

        for neighbour in neighbours:  # For Every Neighbour
            # Get the Neighbour Cell Grid Coordinates
            forward_next_cell = [neighbour[0] + forward_current[0],
                                 neighbour[1] + forward_current[1]]
            # If Neighbour Within Grid
            if forward_next_cell[0] >= 0 and forward_next_cell[0] < len(grid[0]) and forward_next_cell[1] >= 0 and forward_next_cell[1] < len(grid) and forward_next_cell not in closed_queue:
                # If the Neighbour Cell is Not a Wall
                if grid[forward_next_cell[1]][forward_next_cell[0]] != 1:
                    # Add it to the Queue of Nodes to be Searched
                    open_queue.append(forward_next_cell)
                    # Add Current Node to List of Searched Nodes
                    closed_queue.append(forward_next_cell)
                    # Add the Path Taken to the Neighbour Cell to Path
                    graph_tree.append(
                        {"Start": forward_current, "Dest": forward_next_cell})

        return open_queue, closed_queue, graph_tree


    # The method is used to deduce the shortest path from all the cells explored in the grid using BFS Algorithm
    def bfs_path_deduction(self, start, goal,  graph_tree):
        shortest = [goal]
        while goal != start:  # While the Start Pos Hasn't Yet Been Reached
            for step in graph_tree:  # Loop Through Every Node With a path Stored in the graph tree
                # If the BFS Acessed the Current Target From This Node
                if step["Dest"] == goal:
                    # Make the New Current Target the One the Old Target was Acessed From
                    goal = step["Start"]
                    # Insert the Old Target at the Start of the Shortest Path
                    shortest.insert(0, goal)
        return shortest


    # Bidirectional Dijkstra Search
    def BidirectionalDijkstraSearch(self, start, goal):
        # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
        distance = self.Heuristic(start, goal, False)

        # If the Manhattan distance between the start and goal nodes is equal to (adjacent) '1' grid cell (do not generate a path), do the following
        if distance == 1:
            # Return the path as the start and goal nodes only
            return [start, goal]

        # Else if the Manhattan distance between the start and target nodes is larger than '1' grid cell (generate a path), do the following
        else:
            # Define Grid Array
            # Define the Grid to be Searched
            grid = [[0 for x in range(30)] for x in range(28)]
            for cell in self.app.walls:  # Add the Game Walls to the Grid
                if cell.x < 30 and cell.y < 28:
                    grid[int(cell.x)][int(cell.y)] = 1

            #### FORWARD DIRECTION VARIABLES ####
            # Define Node G Score Array
            forward_NodeGScore = [[0 for x in range(30)]
                                  for x in range(28)]
            # Queue for storing the cells for which score has been calculated
            forward_GScoreCheckQueue = [start]
            forward_GScoreCheckQueueIndex = 0
            # The G score or the tentative distance for the starting cell is obviously zero
            forward_NodeGScore[start[0]][start[1]] = 0

            #### BACKWARD DIRECTION VARIABLES ####
            # Define Node G Score Array
            backward_NodeGScore = [[0 for x in range(30)]
                                   for x in range(28)]
            # Queue for storing the cells for which score has been calculated
            backward_GScoreCheckQueue = [goal]
            backward_GScoreCheckQueueIndex = 0
            # The G score or the tentative distance for the starting cell is obviously zero
            backward_NodeGScore[goal[0]][goal[1]] = 0

            # Update the costs associated with the neighbouring node relative to the distance between the start node and the goal node
            distance = self.Heuristic(start, goal, False)

            intersection = []  # Variable for intersecting node/cell between the two explorations

            # Loop until both the Queues (from both the ends) are checked
            while(forward_GScoreCheckQueueIndex < len(forward_GScoreCheckQueue) and backward_GScoreCheckQueueIndex < len(backward_GScoreCheckQueue)):

                # Explore nodes/cells in forward direction (from start to intersecting cell)
                forward_GScoreCheckQueue, forward_GScoreCheckQueueIndex, forward_NodeGScore = self.dijkstra_exploration(
                    forward_GScoreCheckQueue, forward_GScoreCheckQueueIndex, forward_NodeGScore, grid)

                # Explore nodes/cells in backward direction (from target to intersecting cell)
                backward_GScoreCheckQueue, backward_GScoreCheckQueueIndex, backward_NodeGScore = self.dijkstra_exploration(
                    backward_GScoreCheckQueue, backward_GScoreCheckQueueIndex, backward_NodeGScore, grid)
              
                intersection = []  # Variable for intersecting node/cell between the two explorations
                # If there is/are common cells between forward visited cells and backward visited cells,
                # Make those cells intersection cells and break out of the while loop
                for cell in forward_GScoreCheckQueue:
                    if cell in backward_GScoreCheckQueue:
                        intersection.append(cell)
                        break

            # Variable for complete shortest path from start to target
            #complete_path = [target, start]

            # if the queue is not empty
            if len(forward_GScoreCheckQueue) != 0:
                # make intersecting cell/node the target for the forward direction
                forward_target = list(intersection[0])
                # Deduce shortest path from the explored nodes/cells in forward direction till intersecting cell/node
                forward_path = self.dijkstra_path_deduction(
                    start, forward_target, forward_NodeGScore, grid)

            # if the queue is not empty
            if len(backward_GScoreCheckQueue) != 0:
                # make intersecting cell/node the target for the backward direction
                backward_target = list(intersection[0])
                # Deduce shortest path from the explored nodes/cells in backward direction till intersecting cell/node
                backward_path = self.dijkstra_path_deduction(
                    goal, backward_target, backward_NodeGScore, grid)

            backward_path = list(reversed(backward_path))
            complete_path = forward_path + backward_path

            # Return path in reverse direction and shunt out extra elements
            complete_path = list(reversed(complete_path))
            complete_path.pop()
        
            return complete_path


    # Explore the cells in the grid starting from 'start' node to 'goal' node
    def dijkstra_exploration(self, GScoreCheckQueue, GScoreCheckQueueIndex, NodeGScore, grid):
        # Get Current Node
        CurrentNode = GScoreCheckQueue[GScoreCheckQueueIndex]
        # Get current G Score
        CurrentNodeGScore = NodeGScore[CurrentNode[0]][CurrentNode[1]]

        # Define an Array of Current's Neighbours (Offsets)
        neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]

        # Add Every Neighbour to Check Queue
        for neighbour in neighbours:
            # Get the Neighbour Cell Index
            NeighbourCell = [neighbour[0] + CurrentNode[0],
                             neighbour[1] + CurrentNode[1]]  # Get Neighbour Cell
            # If Within Grid Bounds
            if(NeighbourCell[0] >= 0 and NeighbourCell[0] < len(grid) and NeighbourCell[1] >= 0 and NeighbourCell[1] < len(grid[0])):
                if (NeighbourCell not in GScoreCheckQueue):  # If Not Already in Queue
                    if(grid[NeighbourCell[0]][NeighbourCell[1]] != 1):  # If Cell Not a Wall
                        # Append Neighbour Cell to Checked List
                        GScoreCheckQueue.append(NeighbourCell)
                        # Assign G Score to the cell
                        NodeGScore[NeighbourCell[0]
                                   ][NeighbourCell[1]] = CurrentNodeGScore + 1
                    else:
                        # Assign G Score for Wall
                        NodeGScore[NeighbourCell[0]
                                   ][NeighbourCell[1]] = 10000

        # Increment Counter
        GScoreCheckQueueIndex += 1
        return GScoreCheckQueue, GScoreCheckQueueIndex, NodeGScore


    # The method is used to deduce the shortest path from all the cells explored in the grid using Dijkstra's Algorithm
    def dijkstra_path_deduction(self, start, goal, NodeGScore, grid):
        # Define Path to Return
        path = [goal]
        while (goal != start):  # While the Start Pos Hasn't Yet Been Reached
            # Append Neighbour With Lowest G Score
            LowestNeighbour = goal
            targetGScore = NodeGScore[goal[0]][goal[1]]

            # Get the Neighbour Cell Index
            neighbours = [[0, -1], [1, 0], [0, 1], [-1, 0]]
            for neighbour in neighbours:
                NeighbourCell = [neighbour[0] +
                                 goal[0], neighbour[1] + goal[1]]
                # If Within Grid Bounds
                if(NeighbourCell[0] >= 0 and NeighbourCell[0] < len(grid) and NeighbourCell[1] >= 0 and NeighbourCell[1] < len(grid[0])):
                    if (grid[NeighbourCell[0]][NeighbourCell[1]] != 1):  # If Not a Wall
                        if (NodeGScore[NeighbourCell[0]][NeighbourCell[1]] == (targetGScore - 1)):
                            LowestNeighbour = NeighbourCell
             # Append Lowest Neighbour
            path.append(LowestNeighbour)

            # Lowest Neighbour Becomes New Target
            goal = LowestNeighbour
        return path


    # Iterative Testing Functions
    def InitNextTest(self):
        # Stop PacMan Moving
        self.SetDir(0, 0)

        # Output Current Test Results to File
        f = open(self.OutputFile, "a")
        f.write("\n{0}, [{1},{2}], {3}, {4}".format(self.TestItCounter, int(self.starting_pos[0]), int(
            self.starting_pos[1]), self.NodesTraversed, self.app.time_elapsed))
        f.close()

        # Print Current Test Results
        print("{0} Test No: {1}, Start Pos: {2}, Nodes Traversed: {3}, Time Elapsed: {4}".format(
            self.AIType, self.TestItCounter, self.starting_pos, self.NodesTraversed, self.app.time_elapsed))

        # Reset Clock
        self.app.time_elapsed = 0.0
        self.app.time_current = 0.0
        self.app.time_prior = 0.0

        # Increment Test Counter
        self.TestItCounter += 1

        # ResetNodes Traversed Counter
        self.NodesTraversed = 0

        # Test on Next Pos or Reset
        if(self.TestItCounter <= len(self.StartPosArray)):  # If More Tests to Run
            # Move to New Start Pos
            self.starting_pos = [self.StartPosArray[self.TestItCounter -
                                                    1].x, self.StartPosArray[self.TestItCounter - 1].y]
            self.grid_pos = [self.StartPosArray[self.TestItCounter -
                                                1].x, self.StartPosArray[self.TestItCounter - 1].y]
            self.pix_pos = self.get_pix_pos(self.grid_pos[0], self.grid_pos[1])

            # Repopulate Coins
            for i in range(len(self.OriginalCoinsArray)):
                self.app.coins.append(self.OriginalCoinsArray[i])

            # If on a Coin, Eat it
            if self.grid_pos in self.app.coins:
                # Eat the Coin
                self.eat_coin()

            # Get PacMan's Current Position as the Start Pos
            self.start = [int(self.grid_pos[0]), int(self.grid_pos[1])]

            # Use BFS (Search Only Here) to Find the Closest Coin to Pathfind to
            self.target = self.BreadthFirstSearchFindCoin(self.start)

            # Use Relevant Pathfinding to Pathfind to Coin
            self.PathfindUsingSetAI()
        else:  # If All Test Completed
            if (self.bTestAll):  # If Testing All
                # Output
                print("Testing on {0} PacMan Complete".format(
                    self.app.PacManAITypes[self.TestAllAICounter]))

                # Increment Test Counter
                self.TestAllAICounter += 1

                # Check if AI's Left to Test
                if(self.TestAllAICounter < len(self.app.PacManAITypes)):  # If AIs Left to Test
                    # Define New AI
                    self.AIType = self.app.PacManAITypes[self.TestAllAICounter]

                    # Output
                    print("\nCommencing Testing on {0} PacMan:".format(
                        self.AIType))

                    # Define New Output File
                    self.OutputFile = "Results/" + self.AIType + "_Results.txt"

                    # Define New Output File
                    f = open(self.OutputFile, "w")  # Create File
                    # Write Headers
                    f.write("Test, Start_Pos, Nodes_Traversed, Time Elapsed")
                    f.close()  # Close File

                    # Reset Test Counter
                    self.TestItCounter = 1

                    # Move to New Start Pos
                    self.starting_pos = [
                        self.StartPosArray[0].x, self.StartPosArray[0].y]
                    self.grid_pos = [self.StartPosArray[0].x,
                                     self.StartPosArray[0].y]
                    self.pix_pos = self.get_pix_pos(
                        self.grid_pos[0], self.grid_pos[1])

                    # Repopulate Coins
                    for i in range(len(self.OriginalCoinsArray)):
                        self.app.coins.append(self.OriginalCoinsArray[i])

                    # If on a Coin, Eat it
                    if self.grid_pos in self.app.coins:
                        # Eat the Coin
                        self.eat_coin()

                    # Get PacMan's Current Position as the Start Pos
                    self.start = [int(self.grid_pos[0]), int(self.grid_pos[1])]

                    # Use BFS (Search Only Here) to Find the Closest Coin to Pathfind to
                    self.target = self.BreadthFirstSearchFindCoin(self.start)

                    # Use Relevant Pathfinding to Pathfind to Coin
                    self.PathfindUsingSetAI()
                else:  # If Final Test Completed
                    # Output
                    print("All PacMan AIs Tested")
            else:
                print("{0} PacMan Tests Complete".format(self.AIType))


    # General Functions
    # Eat a Collided Coin
    def eat_coin(self):
        self.app.coins.remove(self.grid_pos)
        #self.current_score += 1

        coins_max = len(self.OriginalCoinsArray)
        coins_collected = coins_max - len(self.app.coins)

        completion = round((coins_collected / coins_max) * 100, 2)

        pygame.display.set_caption(self.AIType + " [" + str(completion) + "%]")


    # Move Function Called in App Class to Set Stored Direction to Given Input
    def move(self, direction):
        # Set Direction
        self.stored_direction = direction


    # Set Directiion Function, Used to Localise Setting PacMan Direction and Cap Within Range
    def SetDir(self, DirX, DirY):
        # Cap DirX
        if (DirX > 1):
            DirX = 1
        elif (DirX < -1):
            DirX = -1

        # Cap DirY
        if (DirY > 1):
            DirY = 1
        elif (DirY < -1):
            DirY = -1

        # Set Direction
        self.direction = vec(DirX, DirY)


    # Get Pixel Position
    def get_pix_pos(self, GridX, GridY):
        return vec((GridX*self.app.cell_width)+TOP_BOTTOM_BUFFER//2+self.app.cell_width//2,
                   (GridY*self.app.cell_height) +
                   TOP_BOTTOM_BUFFER//2+self.app.cell_height//2)

        print(self.grid_pos, self.pix_pos)


    # Return if PacMan Has Finished Moving to the Next Node in His Path
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


    # Return if PacMan is Able to Move in Current Direction
    def can_move(self):
        for wall in self.app.walls:
            if vec(self.grid_pos+self.direction) == wall:
                return False
        return True