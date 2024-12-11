"""
This program is for creating and spawning robots within a given
task environment, such that tasks can be allocated to them using
a fuzzy inference system.

This is for testing the fuzzy inference system on fake, simulated 
agents within a fictitious environment

THIS FILE RUNS NUMEROUS SIMULATIONS

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
from PythonFISFunction2 import *
import pandas as pd
import tkinter as tk
import statistics
import time

################# Function & Class Definition ###################

class Robot:
    """
    This is a simplistic robot class for use in the FIS, it is 
    used to create robotic objects. A robot consists of:
    - an ID tag, for referencing
    - a sensor type, either imagery, measurement, or both
    - a load history, which denotes how many times the robot has 
      gone to the task site
    - their position within space
    - a travel distance, which represents how far a robot has to travel to the task site
    - a total travel distance that they have travelled overall
    - a weight, which is used to quantify the impact of their travelling
    - a suitability, which is to be calculated using the FIS
    - a colour used in plotting
    """
    
    # constructor for robot objects:
    def __init__(self, id, sensor, position):
        self.id = id            # id tag for referencing
        self.sensor = sensor    # type of sensor the robot is equipped with
        self.load = 0.0             # load history of robot
        self.position = position    # current position of robot
        self.travel = 0.0       # distance robot must travel to task site
        self.total = 0.0        # total distance that robot has travelled
        self.weight = float(1 + random.uniform(-0.1, 0.1))       # movement weight
        self.suitability = 0.0     # suitability
        self.colour = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))   # colour identifier

    # for querying robots:
    def display_robot_info(self):
        return (f"Robot ID: {self.id}\n"
                f"Position: {self.position}\n"
                f"Sensor Type: {self.sensor}\n"
                f"Load History: {self.load}\n"
                f"Travelled Distance: {self.travel}\n"
                f"Suitability: {self.suitability}")

def read_map(map_str, resolution):

    # get cwd, list all directories and append to the file path of the maps
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)
    file_path = os.path.join(current_dir, files_in_dir[files_in_dir.index('python_stuff')], "maps", str(map_str))

    # check if that map exists, read image if it does
    if not os.path.isfile(file_path):
        sys.exit('No such file exists')
    else: 
        image = cv2.imread(file_path,0)

    # load the map and dilate the borders to get a buffered image for navigation:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    buffered_image, spawn_locations = add_buffer(image, buffer)

    return image_rgb, buffered_image

def dijkstra(image, start, goal):

    # first need to get the dimensionality of the image:
    rows, cols = image.shape

    # swap the (x, y) input to (y, x) format for internal use - cv2 has y,x notation
    start = (start[1], start[0])  # swap (x, y) -> (y, x)
    goal = (goal[1], goal[0])     # swap (x, y) -> (y, x)

    # need to initialize both the distance map and the previous node map:
    dist = np.full((rows,cols), np.inf)  # set other distances to a very big number
    dist[start] = 0                      # set the initial starting distance to 0

    # need to track the parent of each node such that the resulting path can be reconstructed:
    parent = {start: None}

    # start the priority queue to store the distance values in x and y:
    pq = [(0, start)]

    # encode the directions that the robot can move, assuming 8 options of movement at each given
    # node, 45 degree offsets (holonomic movement):
    directions = [
        (-1,0), (1,0), (0,-1), (0,1),    # left, right, down, up
        (-1,-1), (-1,1), (1,-1), (1,1)   # diagonals
    ]

    # define a set for the visited node:
    visited = set()

    while pq:
        current_dist, (x, y) = heapq.heappop(pq)
        
        # if the node has already been visited, skip it
        if (x, y) in visited:
            continue
        visited.add((x, y))
        
        # if we reached the goal, reconstruct the path:
        if (x, y) == goal:
            path = []
            while (x,y) != start:
                path.append((y,x))
                x,y = parent[(x,y)]
            path.append((start[1],start[0]))
            return path[::-1], dist[goal] # reverse path 
        
    # explore neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < rows and 0 <= ny < cols and (nx, ny) not in visited:
                # ignore black borders
                if image[nx, ny] >= 254:  # threshold for white space
                    # calculate the movement cost
                    movement_cost = m.sqrt(2) if dx != 0 and dy != 0 else 1
                    new_dist = current_dist + movement_cost
                    
                    # if a shorter path is found, update the distance and push to pq
                    if new_dist < dist[nx, ny]:
                        dist[nx, ny] = new_dist
                        parent[(nx,ny)] = (x,y)
                        heapq.heappush(pq, (new_dist, (nx, ny)))
    
    return None, None # if goal unreachable

def add_buffer(image, buffer_size):

    # create a binary mask where black (0) and gray (205) areas are marked
    mask = np.where((image == 0) | (image == 205), 1, 0).astype(np.uint8)
    
    # create a kernel for dilation (buffer expansion)
    kernel = np.ones((buffer_size, buffer_size), np.uint8)
    
    # dilate the mask (expand obstacles)
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    
    # create a new image where dilated areas are treated as non-navigable (set to black)
    buffered_image = image.copy()
    buffered_image[dilated_mask == 1] = 0  # set dilated areas to black (0)

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

def draw_circles_on_image(image):

    # for every robot:
    for robot in robots.values(): 
        # extract robot position
        position = robot.position

        # draw circle on position of robot
        cv2.circle(image, (int(position[0]), int(position[1])), 3, robot.colour, -1)
    
    # draw a circle on the task also
    cv2.circle(image, (int(current_task[0]), int(current_task[1])), 3, (255, 0, 0), -1)

    return image

def initialize_plot():
    
    # start a tkinter window
    root = tk.Tk()

    # get screen width & height off of tkinter window, calculate figure height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    dpi = 100
    fig_width_in = (screen_width*0.8)/dpi
    fig_height_in = (screen_height*0.8)/dpi

    # kill tkinter
    root.destroy()

    # make a figure, set the size and position of the figure
    fig = plt.figure(figsize = (fig_width_in, fig_height_in), dpi = dpi)
    ax = fig.add_subplot(1,1,1)
    figure_manager = plt.get_current_fig_manager()
    fig_width = int(fig_width_in * dpi)
    fig_height = int(fig_height_in * dpi)
    x_position = (screen_width - fig_width) // 2
    y_position = (screen_height - fig_height) // 2
    figure_manager.window.wm_geometry(f"{fig_width}x{fig_height}+{x_position}+{y_position}")

    # scatter in the corner for legend 
    plt.scatter(0,0, color = (1, 0, 0), label = 'Task')
    for id, robot in robots.items():
        plt.scatter(0,0, color = (robot.colour[0]/255, robot.colour[1]/255, robot.colour[2]/255), label = robot.id)  
    plt.legend()

#################             Main             ###################

# simulation parameters:
resolution = 0.05               # resolution of the map, slam_toolbox default
map_str = "warehouse_map.png"   # string value of the map name
visualize = False               # whether to view or not
buffer = 5                      # distance in pixels that obstacles should be avoided

nr = 4                      # number of robots in the MRS
x = 2                       # number of camera equipped robots within the MRS
y = nr - x                  # number of measurement equipped robots within the MRS
task_num = 10               # number of task sites
sim_length = 250            # number of times to simulate allocation
robots = {}                 # empty dictionary to hold robot objects once created

bid = np.zeros((nr,3), dtype = object)      # empty array to store robot bids
cumulative_distance = 0                     # cumulative weighted travel distance amongst all robots, initialized

loads = []              # initialize empty list of loads
total_travel = []       # initialize empty list of total distance travelled per robot

# create empty pandas dataframes:

inputs = pd.DataFrame()     # empty data frame for input 
outputs = pd.DataFrame()    # empty data frame for output

# create fuzzy inference rulebase:
rulebase = fis_create()

# initialize map:
image_rgb, buffered_image = read_map(map_str, resolution)

begin = time.time()

for i in range(0,sim_length):
    nr = 4                      # number of robots in the MRS
    x = 2                       # number of camera equipped robots within the MRS
    y = nr - x                  # number of measurement equipped robots within the MRS

    # determine task and robot sites: 
    match map_str:
    # for the warehouse case:
        case "warehouse_map.png":
            locations = [(64,63), (64,157), (64,231), (63,300), (175, 65), (172,123), 
                (171,188), (180,262), (182,330), (288,74), (221,120), (238,192),
                (229,260), (241,321), (405,324), (405, 105), (339,146), (342,248),
                (285,265), (425,98), (422,150), (417,226), (428,266), (490,304),
                (539,332), (559,290), (591,327),(558,259), (633,333), (481,115),
                (544,115), (604,115), (487,179), (547,179), (603,179), (600,67)]
            random.shuffle(locations)   # shuffle the locations list so that each time the simulation is ran the robots and tasks are in different locations
            tasks = locations[0:task_num]
            positions = locations[task_num::]

    # for the lab case:    
        case "edited_map.png":
            locations = [(36,297), (29,276), (59,300), (73,293), (82,266), (121,278),
                        (106,287), (46,229), (83,232), (102,204), (80,183), (50,174),
                        (77,151), (125,159), (121,104), (118,82), (80,76), (77,49),
                        (76,26), (121,20), (145,23), (185,23), (213,29), (217,66),
                        (183,68), (153,66), (219,86), (235,74), (257,96), (284,80),
                        (63,90), (187,63), (159,11), (71,214), (75,143), (49,294)]
            random.shuffle(locations)   # shuffle the locations list so that each time the simulation is ran the robots and tasks are in different locations
            tasks = locations[0:task_num]
            positions = locations[task_num::]  
  
    # spawn robots based on the user defined mission parameters:
    for num in range(1, nr+1):
        robot_name = f"Robot {num}"

        # if we haven't exceeded the number of imagery robots
        if num <= x:
            robots[robot_name] = Robot(
                id = num,
                sensor = "Imagery",
                position = positions[num-1],
            )
        # else make measurement robots    
        else:
            robots[robot_name] = Robot(
                id = num,
                sensor = "Measurement",
                position = positions[num-1],
            )

    for current_task in tasks:

        # draw the markers for the initial positions of everything
        combined_image = draw_circles_on_image(image_rgb.copy())
        if visualize == True:
            plt.imshow(combined_image)
            plt.draw()
            plt.pause(0.5)

        # query robots and determine suitability:
        for id, robot in robots.items():

            # determine the robots starting position:
            start = robot.position

            # determine the length of the planned path:
            shortest_path, dist = dijkstra(buffered_image, start, current_task)
            
            # add the path if it exists:
            if shortest_path is not None:
                for x,y in shortest_path:
                    combined_image[y,x] = robot.colour

            # update the robots planned travel distance:
            robot.travel = round((dist * resolution),3)

            # need to use the fuzzy inference system to determine the suitability
            # of a given robot for the task:
            robot.suitability = round(fis_solve(rulebase, robot.load, robot.travel, robot.total), 2)

            # fill out the bid array:
            bid[robot.id - 1, 0] = robot.sensor
            bid[robot.id - 1, 1] = robot.suitability
            bid[robot.id - 1, 2] = robot.id

        # re draw with the path:
        if visualize == True:
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

        # save the data that was used to make the decision:

        input_data = [
        {'Load History': robot.load, 'Distance to Task': robot.travel, 'Total Distance Traveled': robot.total}
        for robot in robots.values()    
        ]

        input_df = pd.DataFrame(input_data)
        inputs = pd.concat([inputs, input_df], ignore_index=True)

        output_data = [
            {'Suitability': robot.suitability}
            for robot in robots.values()
        ]

        output_df = pd.DataFrame(output_data)
        outputs = pd.concat([outputs, output_df], ignore_index=True)
    
        # these robots have been selected, send them to the task site and update:
        for id, robot in robots.items():
            if robot.id == imagery_selected[2] or robot.id == measurement_selected[2]:

                # increment the load history of the robot:
                robot.load += 1

                # randomly update the robot position to within the task location:
                robot.position = (current_task[0] + random.randint(-7,7), current_task[1] + random.randint(-7,7))

                # increment the robots individual total travel distance:
                robot.total += robot.travel

                # keep track of the total distance that all robots have travelled:
                cumulative_distance += robot.travel

        # print robot data in terminal:
        robots_data = [
        {'Robot ID': robot.id, 'Sensor Type': robot.sensor, 'Load History': robot.load , 'Distance to Task': robot.travel, 'Total Distance Travelled': robot.total,
        'Suitability': robot.suitability}
        for robot in robots.values()
        ]

        df = pd.DataFrame(robots_data)

        # draw after positions have been updated:
        combined_image = draw_circles_on_image(image_rgb.copy())
        if visualize == True:
            print(df.to_string(index = False, justify = 'center'))
            plt.imshow(combined_image)
            plt.draw()
            plt.pause(1)

    # get metrics after simulation:

    loads.append(df['Load History'].tolist())
    total_travel.append(df['Total Distance Travelled'].tolist())
    print(f"simulation {i+1}/{sim_length}")

end = time.time()

inputs.to_csv('inputs.csv', index = False)
outputs.to_csv('outputs.csv', index = False)

loads = np.array(loads)
total_travel = np.array(total_travel)
print(f"Standard Deviation of Load History: {round(loads.std(),3)}\n"
      f"Average Total Distance: {round(total_travel.mean(),2)}m\n"
      f"Standard Deviation of Total Distance: {round(total_travel.std(),3)}m\n"
      f"Elapsed Time: {round(((end - begin)/60),2)} minutes")
