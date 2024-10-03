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

################ FIS Step 1: Define Fuzzy Sets ##################

"""
Must firstly define the input linguistic variables. This involves:
    - Defining the linguistic variable itself
    - Defining the crisp universe of discourse for this variable
    - Defining the linguistic values and their membership functions
"""

# universes of discourse:

lh_range = np.arange(0,11,1)    # crisp values from 0 to 10
wtd_range = np.arange(0,51,1)   # crisp values from 0 to 50
cap_range = np.arange(0,3,1)    # crisp values from 0 to 2

# define linguistic input variables:

lh = ctrl.Antecedent(lh_range, 'Load History')
wtd = ctrl.Antecedent(wtd_range, 'Weighted Travel Distance')
cap = ctrl.Antecedent(cap_range, 'Capabilities')

# define membership functions for the input variables:
#   - we have 3 linguistic terms for each variable

lh['Low'] = fuzz.trimf(lh.universe, [0, 0, 6])
lh['Medium'] = fuzz.trimf(lh.universe, [1, 5, 9])
lh['High'] = fuzz.trimf(lh.universe, [4, 10, 10])

wtd['Low'] = fuzz.trimf(wtd.universe, [0, 0, 30])
wtd['Medium'] = fuzz.trimf(wtd.universe, [4, 25, 46])
wtd['High'] = fuzz.trimf(wtd.universe, [20, 50, 50])

cap['No Matches'] = fuzz.trimf(cap.universe, [0, 0, 0])
cap['One Match'] = fuzz.trimf(cap.universe, [1, 1, 1])
cap['Two Matches'] = fuzz.trimf(cap.universe, [2, 2, 2])

"""
Now we can define the output linguistic variable. This involves:
    - Defining the linguistic variable itself
    - Defining the crisp universe of discourse for this variable
    - Defining the linguistic values and their membership functions
"""

# universe of discourse:

suit_range = np.arange(0,11,1)

# define linguistic output variable:

suit = ctrl.Consequent(suit_range, 'Suitability')

# membership functions for linguistic values:

suit['Very Low'] = fuzz.trimf(suit_range, [0, 0, 2 + (1/12)])
suit['Low'] = fuzz.trimf(suit_range, [(5/12), 2.5, 4 + (7/12)])
suit['Medium'] = fuzz.trimf(suit_range, [2 + (11/12), 5, 7 + (1/12)])
suit['High'] = fuzz.trimf(suit_range, [5 + (5/12), 7.5, 9 + (7/12)])
suit['Very High'] = fuzz.trimf(suit_range, [7 + (11/12), 10, 10])

################ FIS Step 2: Define Rule-Base ###################

"""
Now we can define the fuzzy rule base. For a system with 3
linguistic inputs, each with 3 linguistic variables, the 
rule-base can contain a maximum of 27 rules for a full
description

The following rules were selected based on their provided
surface of control, which was sculpted using these rules:

"""

# define rules for the mismatched case:

rule1 = ctrl.Rule(cap['No Matches'], suit['Very Low'])

# define rules for the one-match case:

rule2 = ctrl.Rule()
