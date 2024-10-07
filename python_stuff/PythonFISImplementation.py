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

def load_map(map_str, resolution):
    """
    This function reads maps and determines the white space and the border
    based on a provided map and resolution in m/pixel
    """

    # determine if the provided map exists:
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)
    file_path = os.path.join(current_dir, files_in_dir[files_in_dir.index('python_stuff')], "maps", map_str)

    if not os.path.isfile(file_path):
        sys.exit('No such file exists')
    else: 
        image = cv2.imread(file_path,0)

    # determine the white space and the border:

    body = resolution*np.flip(np.column_stack(np.where(np.flipud(image) >= 254)), axis = 1)
    border = resolution*np.flip(np.column_stack(np.where(np.flipud(image) == 0)), axis = 1)
    return body, border, image # return the body and border positions in meters, as well as all the pixel locations

body, border, image = load_map("test_map.png", 0.05)

# for i in image if 205 set as 0 if 0 set as 1 to determine where obstacles are