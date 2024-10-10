"""
This program is for creating and spawning robots within a given
task environment, such that tasks can be allocated to them using
a fuzzy inference system.

This is for testing the fuzzy inference system on fake, simulated 
agents within a fictitious environment

"""
######################## Import Packages ########################

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import sys
import random
import math as m
import heapq
from PythonFISFunctions import *

################# Function & Class Definition ###################

class Robot:
    """
    This is a simplistic robot class for use in the FIS, it is 
    used to create robotic objects. A robot consists of:
    - an ID tag, for referencing
    - a load history, which denotes how many times the robot has 
      gone to the task site
    - a travel distance, which represents how far a robot has traveled
    - a sensor type, either imagery, measurement, or both
    - their position within space
    - a weight, which is used to resolution the impact of their travelling
    - a suitability, which is to be calculated using the FIS
    """
    
    def __init__(self, id, sensor, position):
        self.id = id
        self.sensor = sensor
        self.load = 0
        self.position = position
        self.travel = 0
        self.weight = 1 + random.uniform(-0.01, 0.01)
        self.suitability = 0

    # for querying: 

    def display_robot_info(self):
        return (f"Robot ID: {self.id}\n"
                f"Sensor Type: {self.sensor}\n"
                f"Load History: {self.load}\n"
                f"Travelled Distance: {self.travel}\n"
                f"Suitability: {self.suitability}")

def read_map(map_str, resolution):

    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)
    file_path = os.path.join(current_dir, files_in_dir[files_in_dir.index('python_stuff')], "maps", str(map_str))

    if not os.path.isfile(file_path):
        sys.exit('No such file exists')
    else: 
        image = cv2.imread(file_path,0)

    return image

def dijkstra(image, start, goal):
    # first need to get the dimensionality of the image:

    rows, cols = image.shape

    # Swap the (x, y) input to (y, x) format for internal use
    start = (start[1], start[0])  # Swap (x, y) -> (y, x)
    goal = (goal[1], goal[0])     # Swap (x, y) -> (y, x)

    # need to initialize both the distance map and the previous node map:

    dist = np.full((rows,cols), np.inf)  # set other distances to a very big number
    dist[start] = 0                 # set the initial starting distance to 0

    # need to track the parent of each node such that the resulting path can be reconstructed:

    parent = {start: None}

    # start the priority queue to store the distance values in x and y:

    pq = [(0, start)]

    # encode the directions that the robot can move, assuming a 8 options of movement at each given
    # node (holonomic movement):

    directions = [
        (-1,0), (1,0), (0,-1), (0,1),    # left, right, down, up
        (-1,-1), (-1,1), (1,-1), (1,1)   # diagonals
    ]

    # define a set for the visited node:

    visited = set()

    while pq:
        current_dist, (x, y) = heapq.heappop(pq)
        
        # If the node has already been visited, skip it
        if (x, y) in visited:
            continue
        visited.add((x, y))
        
        # If we reached the goal, reconstruct the path:
        if (x, y) == goal:
            path = []
            while (x,y) != start:
                path.append((y,x))
                x,y = parent[(x,y)]
            path.append((start[1],start[0]))
            return path[::-1], dist[goal] # reverse path 
        
    # Explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                # Ignore black borders (assuming 0 is black, adjust threshold as needed)
                if image[nx, ny] >= 254:  # Change this threshold based on your image
                    # Calculate the movement cost
                    movement_cost = m.sqrt(2) if dx != 0 and dy != 0 else 1
                    new_dist = current_dist + movement_cost
                    
                    # If a shorter path is found, update the distance and push to pq
                    if new_dist < dist[nx, ny]:
                        dist[nx, ny] = new_dist
                        parent[(nx,ny)] = (x,y)
                        heapq.heappush(pq, (new_dist, (nx, ny)))
    
    return None, None # if goal unreachable

def add_buffer(image, buffer_size):
    # Create a binary mask where black (0) and gray (205) areas are marked
    mask = np.where((image == 0) | (image == 205), 1, 0).astype(np.uint8)
    
    # Create a kernel for dilation (buffer expansion)
    kernel = np.ones((buffer_size, buffer_size), np.uint8)
    
    # Dilate the mask (expand obstacles)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Create a new image where dilated areas are treated as non-navigable (set to black)
    buffered_image = image.copy()
    buffered_image[dilated_mask == 1] = 0  # Set dilated areas to black (0)

     # white space detection:
    spawn_locations = np.flip(np.column_stack(np.where(np.flipud(buffered_image) >= 254)),axis = 1)
    
    return buffered_image, spawn_locations

def generate_image(width, height):
    # generate a blank png for use in mapping:
    blank_image = np.ones((height, width, 3), dtype=np.uint8) * 205

    # specify path:
    files_in_dir = os.listdir(os.getcwd())
    file_path = os.path.join(os.getcwd(), files_in_dir[files_in_dir.index('python_stuff')], "maps")

    # write the image to variable that will return a flag for true or false
    a = cv2.imwrite(os.path.join(file_path, 'blank_image.png'), blank_image)

    # verify that the image was actually created:
    if a == True:
        print('Image saved successfully')
    else:
        print('Image saving failed')

#################             Main             ###################

resolution = 0.05
map_str = "warehouse_map.png"
buffer = 5

image = read_map(map_str, resolution)
buffered_image, spawn_locations = add_buffer(image, buffer)

# for robot simulation:
#   - spawn x many robots within set positions, 
#   - randomly spawn a task site
#   - use the fuzzy inference system to determine who is most suitable
#   - allocate the task to the most suitable robot
#   - update robot params and repeat

# start = np.array((410,317))
# end = np.array((619,75))

# shortest_path, distance = dijkstra(buffered_image, start, end)

# if shortest_path is not None:
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

#     for x,y in shortest_path:
#         image_rgb[y,x] = [255, 0, 0]
        
#     print(f'Path length is {round((distance*resolution),2)} m')
#     plt.imshow(image_rgb)
#     plt.show()
# else:
#     print("Goal is unreachable.")
#     plt.imshow(image, cmap = "gray")
#     plt.show()