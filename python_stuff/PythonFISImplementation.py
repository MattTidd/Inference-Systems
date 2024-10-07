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
    - a weight, which is used to scale the impact of their travelling
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




# test:

robot1 = Robot(
    id = 1,
    sensor = "Camera",
    position = [0,0]
)
