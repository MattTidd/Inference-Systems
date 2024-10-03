"""

This program is a Python implementation of a Fuzzy Inference System (FIS)
that was first designed within MATLAB, using the Fuzzy Logic Designer.

This FIS is then going to be ported into Python for use in MRTA, where it will eventually 
become a ROS2 package.

"""
######################## Import Packages ########################

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

####################### Define FIS Steps ########################

"""
Must firstly define the input linguistic variables. This involves:
    - Defining the linguistic variable itself
    - Defining the crisp universe of discourse for this variable
    - Defining the linguistic values and their membership functions
"""

# universe of discourse:

lh_range = np.arange(0,11,1)
print(lh_range)
