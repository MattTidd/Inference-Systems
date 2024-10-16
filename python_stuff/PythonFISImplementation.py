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
from matplotlib.animation import FuncAnimation
import cv2
import os
import sys
import random
from random import shuffle
import math as m
import heapq
from PythonFISFunctions import *
import time


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
    - a color used in plotting
    """
    
    def __init__(self, id, sensor, position):
        self.id = id
        self.sensor = sensor
        self.load = 0.0
        self.position = position
        self.travel = 0.0
        self.weight = float(1 + random.uniform(-0.1, 0.1))
        self.suitability = 0.0
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    # for querying: 

    def display_robot_info(self):
        return (f"Robot ID: {self.id}\n"
                f"Position: {self.position}\n"
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

# for robot simulation:
#   - spawn x many robots within set positions, 
#   - randomly spawn a task site
#   - use the fuzzy inference system to determine who is most suitable
#   - allocate the task to the most suitable robot
#   - update robot params and repeat

# simulation parameters:
resolution = 0.05               # resolution of the map, slam_toolbox default
map_str = "warehouse_map.png"   # string value of the map name
buffer = 5                      # distance in pixels that obstacles should be avoided

nr = 4                      # number of robots in the MRS
x = 2                       # number of camera equipped robots within the MRS
y = nr - x                  # number of measurement equipped robots within the MRS
robots = {}                 # empty dictionary to hold robot objects once created

bid = np.zeros((nr,3), dtype = object)      # empty array to store robot bids
total_travel = 0                            # total weighted travel distance, initialized

positions = [(410, 317), (601, 117), (152, 329), (239, 70)]
tasks = [(540, 263),(106, 271), (65, 63), (602, 196), (401, 190),
         (401, 190), (313, 79), (487, 229)]
    
# load the map and dilate the borders to get a buffered image for navigation:
image = read_map(map_str, resolution)
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
buffered_image, spawn_locations = add_buffer(image, buffer)

# shuffle indices of positions and tasks:
random.shuffle(positions)   # shuffle the positions of the robots so that each time the simulation is ran the robots are in different locations
random.shuffle(tasks)       # shuffle the ordering of the task locations so that each time the simulation is ran the task ordering is different

# spawn robots based on the user defined mission parameters:
for num in range(1, nr+1):
    robot_name = f"Robot {num}"

    if num <= x:
        robots[robot_name] = Robot(
            id = num,
            sensor = "Imagery",
            position = positions[num-1],
        )
    else:
        robots[robot_name] = Robot(
            id = num,
            sensor = "Measurement",
            position = positions[num-1],
        )

# create fuzzy inference rulebase:
rulebase = fis_create()

def draw_circles_on_image(image):
    for robot in robots.values(): 
        position = robot.position
        cv2.circle(image, (int(position[0]), int(position[1])), 3, robot.color, -1)  
    cv2.circle(image, (int(current_task[0]), int(current_task[1])), 3, (255, 0, 0), -1)
    return image

for current_task in tasks:

    combined_image = draw_circles_on_image(image_rgb.copy())
    plt.imshow(combined_image)
    plt.draw()
    plt.pause(0.5)

    # query robots and determine suitability:

    for id, robot in robots.items():
        # determine the robots starting position:
        start = robot.position

        # determine the length of the planned path:
        shortest_path, dist = dijkstra(buffered_image, start, current_task)
        
        if shortest_path is not None:
            for x,y in shortest_path:
                combined_image[y,x] = robot.color

        # update the robots planned weighted travel distance:
        robot.travel = round((robot.weight * dist * resolution),3)

        # determine the capability matching of the robot:

        check = robot.sensor

        capability = 2 if check in ["Imagery and Measurement", "Measurement and Imagery"] else 1 if check in ["Imagery", "Measurement"] else 0

        # need to use the fuzzy inference system to determine the suitability
        # of a given robot for the task:

        robot.suitability = round(fis_solve(rulebase, robot.load, robot.travel, capability),2)

        # fill out the bid array:
        bid[robot.id - 1, 0] = robot.sensor
        bid[robot.id - 1, 1] = robot.suitability
        bid[robot.id - 1, 2] = robot.id

    plt.imshow(combined_image)
    plt.draw()
    plt.pause(1)

    # sort the bids by highest to lowest suitability:
    sorted_arr = bid[bid[:, 1].astype(float).argsort()[::-1]]

    imagery_selected = None
    measurement_selected = None

    # choose the highest suitability for both capability types:

    for row in sorted_arr:
        if row[0] == 'Imagery' and imagery_selected is None:
            imagery_selected = row
        elif row[0] == 'Measurement' and measurement_selected is None:
            measurement_selected = row
    
        if imagery_selected is not None and measurement_selected is not None:
            break
   
   # these robots have been selected, send them to the task site and update:

    for id, robot in robots.items():
        if robot.id == imagery_selected[2] or robot.id == measurement_selected[2]:
            robot.load += 1
            # randomly update the robot position to within the task location:
            robot.position = (current_task[0] + random.randint(-7,7), current_task[1] + random.randint(-7,7))
            total_travel += robot.travel

    combined_image = draw_circles_on_image(image_rgb.copy())
    plt.imshow(combined_image)
    plt.draw()
    plt.pause(1)
