"""

This program is a Python implementation of a Fuzzy Inference System (FIS)
that was first designed within MATLAB, using the Fuzzy Logic Designer.

This file serves to host the functions that are used within the FIS testing.

This particular FIS utilizes the load history, weighted travel distance to task,
and the capabilities of each robot expressed as a triangular singleton membership
function.

"""
######################## Import Packages ########################

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def fis_create():

    ################ FIS Step 1: Define Fuzzy Sets ##################

    """
    Must firstly define the input linguistic variables. This involves:
        - Defining the linguistic variable itself
        - Defining the crisp universe of discourse for these variables
        - Defining the linguistic values and their membership functions
    """
    # universes of discourse:

    lh_range = [0, (5/6), 4, 5, 6, (55/6), 10]          # crisp values of importance from 0 to 10
    wtd_range = [0, (25/6), 20, 25, 30, (271/6), 50]    # crisp values of importance from 0 to 50
    cap_range = [0, 1, 2]                               # crisp values of importance from 0 to 2

    # define linguistic input variables:

    lh = ctrl.Antecedent(lh_range, 'Load History')
    wtd = ctrl.Antecedent(wtd_range, 'Weighted Travel Distance')
    cap = ctrl.Antecedent(cap_range, 'Capabilities')

    # define membership functions for the input variables:
    #   - we have 3 linguistic terms for each variable

    lh['Low'] = fuzz.trimf(lh.universe, [0, 0, 6])
    lh['Medium'] = fuzz.trimf(lh.universe, [5/6, 5, 55/6])
    lh['High'] = fuzz.trimf(lh.universe, [4, 10, 10])

    wtd['Low'] = fuzz.trimf(wtd.universe, [0, 0, 30])
    wtd['Medium'] = fuzz.trimf(wtd.universe, [25/6, 25, 271/6])
    wtd['High'] = fuzz.trimf(wtd.universe, [20, 50, 50])

    cap['No Matches'] = fuzz.trimf(cap.universe, [0, 0, 0])
    cap['One Match'] = fuzz.trimf(cap.universe, [1, 1, 1])
    cap['Two Matches'] = fuzz.trimf(cap.universe, [2, 2, 2])

    """
    Now we can define the output linguistic variable. This involves:
        - Defining the linguistic variable itself
        - Defining the crisp universe of discourse for this variable
        - Defining the linguistic value and its membership function
    """

    # universe of discourse:

    suit_range = [0, (5/12), (25/12), 2.5, (35/12), (55/12), 5, (65/12), (85/12), 7.5, (95/12), (115/12), 10]

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

    """
    rulebase = []

    # define rules for the mismatched case:

    rulebase.append(ctrl.Rule(cap['No Matches'], suit['Very Low']))

    # define rules for the one-match case:

    rulebase.append(ctrl.Rule(lh['Low'] & wtd['Low'] & cap['One Match'], suit['High']))
    rulebase.append(ctrl.Rule(lh['Medium'] & wtd['Low'] & cap['One Match'], suit['Medium']))
    rulebase.append(ctrl.Rule(lh['High'] & wtd['Low'] & cap['One Match'], suit['Medium']))
    rulebase.append(ctrl.Rule(lh['Low'] & wtd['Medium'] & cap['One Match'], suit['Medium']))
    rulebase.append(ctrl.Rule(lh['Medium'] & wtd['Medium'] & cap['One Match'], suit['Low']))
    rulebase.append(ctrl.Rule(lh['High'] & wtd['Medium'] & cap['One Match'], suit['Low']))
    rulebase.append(ctrl.Rule(lh['Low'] & wtd['High'] & cap['One Match'], suit['Medium']))
    rulebase.append(ctrl.Rule(lh['Medium'] & wtd['High'] & cap['One Match'], suit['Low']))
    rulebase.append(ctrl.Rule(lh['High'] & wtd['High'] & cap['One Match'], suit['Very Low']))

    # define rules for the two-match case:

    rulebase.append(ctrl.Rule(lh['Low'] & wtd['Low'] & cap['Two Matches'], suit['Very High']))
    rulebase.append(ctrl.Rule(lh['Medium'] & wtd['Low'] & cap['Two Matches'], suit['High']))
    rulebase.append(ctrl.Rule(lh['High'] & wtd['Low'] & cap['Two Matches'], suit['High']))
    rulebase.append(ctrl.Rule(lh['Low'] & wtd['Medium'] & cap['Two Matches'], suit['High']))
    rulebase.append(ctrl.Rule(lh['Medium'] & wtd['Medium'] & cap['Two Matches'], suit['Medium']))
    rulebase.append(ctrl.Rule(lh['High'] & wtd['Medium'] & cap['Two Matches'], suit['Medium']))
    rulebase.append(ctrl.Rule(lh['Low'] & wtd['High'] & cap['Two Matches'], suit['High']))
    rulebase.append(ctrl.Rule(lh['Medium'] & wtd['High'] & cap['Two Matches'], suit['Medium']))
    rulebase.append(ctrl.Rule(lh['High'] & wtd['High'] & cap['Two Matches'], suit['Very Low']))

    # define sculpting rules:

    rulebase.append(ctrl.Rule(lh['High'] & cap['One Match'], suit['Very Low']))
    rulebase.append(ctrl.Rule(wtd['High'] & cap['One Match'], suit['Very Low']))

    rulebase.append(ctrl.Rule(lh['High'] & cap['Two Matches'], suit['Very Low']))
    rulebase.append(ctrl.Rule(wtd['High'] & cap['Two Matches'], suit['Very Low']))

    return rulebase

def fis_solve(rulebase, load, travel, capability):

    """
    We first create the control system, and we can run simulations on this control system
    by further passing inputs into it, and then getting it to compute the output.

    """

    # create control system:

    fis_ctrl = ctrl.ControlSystem(rulebase)

    # create an instance of the control system for simulation:

    sim = ctrl.ControlSystemSimulation(fis_ctrl)

    # solve:

    sim.input['Load History'] = load
    sim.input['Weighted Travel Distance'] = travel
    sim.input['Capabilities'] = capability

    sim.compute()

    result = sim.output['Suitability']
    return result