"""

This program is a Python implementation of a Fuzzy Inference System (FIS)
that was first designed within MATLAB, using the Fuzzy Logic Designer.

This file serves to host the functions that are used within the FIS testing.

This particular FIS utilizes the load history, weighted travel distance to task,
and the distance history to make decisions about the suitability of a given robot

"""
######################## Import Packages ########################

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

####################### Define Functions ########################

def fis_create():   

    ################ FIS Step 1: Define Fuzzy Sets ##################

    """
    Must firstly define the input linguistic variables. This involves:
        - Defining the linguistic variable itself
        - Defining the crisp universe of discourse for these variables
        - Defining the linguistic values and their membership functions
    """
    # universes of discourse:

    lh_range = [0, (5/6), 4, 5, 6, (55/6), 10]
    dtt_range = [0, (25/12), 10, 12.5, 15, (275/12), 25]
    tdt_range = [0, (25/6), 15, 25, 30, (275/6), 50]

    # define lingusitic input variables:

    lh = ctrl.Antecedent(lh_range, 'Load History')
    dtt = ctrl.Antecedent(dtt_range, 'Distance to Task')
    tdt = ctrl.Antecedent(tdt_range, 'Total Distance Travelled')

    # define membership functions:
    #   - we have 3 linguistic terms for each variable

    lh['Low'] = fuzz.trimf(lh.universe, [0, 0, 6])
    lh['Medium'] = fuzz.trimf(lh.universe, [5/6, 5, 55/6])
    lh['High'] = fuzz.trimf(lh.universe, [4, 10, 10])

    dtt['Low'] = fuzz.trimf(dtt.universe, [0, 0, 15])
    dtt['Medium'] = fuzz.trimf(dtt.universe, [25/12, 12.5, 275/12])
    dtt['High'] = fuzz.trimf(dtt.universe, [10, 25, 25])

    tdt['Low'] = fuzz.trimf(tdt.universe, [0, 0, 30])
    tdt['Medium'] = fuzz.trimf(tdt.universe, [25/6, 25, 275/6])
    tdt['High'] = fuzz.trimf(tdt.universe, [15, 50, 50])

    """
    Now we can define the output linguistic variable. This involves:
        - Defining the linguistic variable itself
        - Defining the crisp universe of discourse for this variable
        - Defining the linguistic value and its membership function
    """

    # universe of discourse:

    suit_range = [0, (5/12), (25/12), 2.5, (35/12), (55/12), 5, (65/12), (85/12), 7.5, (95/12), (115/12), 10]



   