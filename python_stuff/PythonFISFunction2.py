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

    suit_range = [0, 5/12, 25/12, 2.5, 35/12, 55/12, 5, 65/12, 85/12, 7.5, 95/12, 115/12, 10]

    # define linguistic output variable:

    suit = ctrl.Consequent(suit_range, 'Suitability')

    # membership functions for linguistic values:

    suit['Very Low'] = fuzz.trimf(suit.universe, [0, 0, 25/12])
    suit['Low'] = fuzz.trimf(suit.universe, [5/12, 2.5, 55/12])
    suit['Medium'] = fuzz.trimf(suit.universe, [35/12, 5, 85/12])
    suit['High'] = fuzz.trimf(suit.universe, [65/12, 7.5, 115/12])
    suit['Very High'] = fuzz.trimf(suit.universe, [95/12, 10, 10])

    ################ FIS Step 2: Define Rule-Base ###################

    """
    Now we can define the fuzzy rule base. For a system with 3
    linguistic inputs, each with 3 linguistic variables, the 
    rule-base can contain a maximum of 27 rules for a full
    description

    The following rules were selected based on their provided
    surface of control, which was sculpted iteratively through
    the rules.

    This rule base consists of 27 primary rules and 5 secondary 
    sculpting rules

    """

    rulebase = []

    # commence defining the main rules:

    rulebase.append(ctrl.rule(lh['Low'] & dtt['Low'] & tdt['Low'], suit['High']))           # rule 01
    rulebase.append(ctrl.rule(lh['Medium'] & dtt['Low'] & tdt['Low'], suit['High']))        # rule 02
    rulebase.append(ctrl.rule(lh['High'] & dtt['Low'] & tdt['Low'], suit['Medium']))        # rule 03
    rulebase.append(ctrl.rule(lh['Low'] & dtt['Medium'] & tdt['Low'], suit['High']))        # rule 04
    rulebase.append(ctrl.rule(lh['Medium'] & dtt['Medium'] & tdt['Low'], suit['Medium']))   # rule 05
    rulebase.append(ctrl.rule(lh['High'] & dtt['Medium'] & tdt['Low'], suit['Medium']))     # rule 06
    rulebase.append(ctrl.rule(lh['Low'] & dtt['High'] & tdt['Low'], suit['Medium']))        # rule 07
    rulebase.append(ctrl.rule(lh['Medium'] & dtt['High'] & tdt['Low'], suit['Medium']))     # rule 08
    rulebase.append(ctrl.rule(lh['High'] & dtt['High'] & tdt['Low'], suit['Low']))          # rule 09
    rulebase.append(ctrl.rule(lh['Low'] & dtt['Low'] & tdt['Medium'], suit['High']))        # rule 10
    rulebase.append(ctrl.rule(lh['Medium'] & dtt['Low'] & tdt['Medium'], suit['Medium']))   # rule 11
    rulebase.append(ctrl.rule(lh['High'] & dtt['Low'] & tdt['Medium'], suit['High']))       # rule 12
    rulebase.append(ctrl.rule(lh['Low'] & dtt['Medium'] & tdt['Medium'], suit['Medium']))   # rule 13
    rulebase.append(ctrl.rule(lh['Medium'] & dtt['Medium'] & tdt['Medium'], suit['Low']))   # rule 14
    rulebase.append(ctrl.rule(lh['High'] & dtt['Medium'] & tdt['Medium'], suit['Low']))     # rule 15
    rulebase.append(ctrl.rule(lh['Low'] & dtt['High'] & tdt['Medium'], suit['Medium']))     # rule 16
    rulebase.append(ctrl.rule(lh['Medium'] & dtt['High'] & tdt['Medium'], suit['Low']))     # rule 17
    rulebase.append(ctrl.rule(lh['High'] & dtt['High'] & tdt['Medium'], suit['Very Low']))  # rule 18
    rulebase.append(ctrl.rule(lh['Low'] & dtt['Low'] & tdt['High'], suit['High']))
    rulebase.append(ctrl.rule(lh['Low'] & dtt['Low'] & tdt['Low'], suit['High']))
    rulebase.append(ctrl.rule(lh['Low'] & dtt['Low'] & tdt['Low'], suit['High']))
    rulebase.append(ctrl.rule(lh['Low'] & dtt['Low'] & tdt['Low'], suit['High']))
    rulebase.append(ctrl.rule(lh['Low'] & dtt['Low'] & tdt['Low'], suit['High']))
    rulebase.append(ctrl.rule(lh['Low'] & dtt['Low'] & tdt['Low'], suit['High']))


    # commit after finishing function, commit when starting solver function, 
    # commit when finishing staring function


   