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
from skfuzzy import membership as mem
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

###################### Function Definition ######################

def fis(load, travel, capability, mode):

    ################ FIS Step 1: Define Fuzzy Sets ##################

    """
    Must firstly define the input linguistic variables. This involves:
        - Defining the linguistic variable itself
        - Defining the crisp universe of discourse for these variables
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
        - Defining the linguistic value and its membership function
    """

    # universe of discourse:

    suit_range = np.arange(0,11,1)

    # define linguistic output variable:

    suit = ctrl.Consequent(suit_range, 'Suitability')

    # membership functions for linguistic values:

    suit['Very Low'] = fuzz.trimf(suit_range, [0, 0, 2])
    suit['Low'] = fuzz.trimf(suit_range, [(5/12), 2.5, 4 + (7/12)])
    suit['Medium'] = fuzz.trimf(suit_range, [3, 5, 7])
    suit['High'] = fuzz.trimf(suit_range, [5 + (5/12), 7.5, 9 + (7/12)])
    suit['Very High'] = fuzz.trimf(suit_range, [8, 10, 10])

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

    ############# FIS Step 3: Control System Creation ###############

    """
    We first create the control system, and we can run simulations on this control system
    by further passing inputs into it, and then getting it to compute the output.

    """

    # create the control system:

    fis_ctrl = ctrl.ControlSystem(rulebase)

    # create an instance of the control system for simulation:

    fis_sim = ctrl.ControlSystemSimulation(fis_ctrl, flush_after_run = 50 * 50 + 1)

    # if the user just wants to use the fis:

    if mode == 'Compute':

        fis_sim.input['Load History'] = load
        fis_sim.input['Weighted Travel Distance'] = travel
        fis_sim.input['Capabilities'] = capability

        fis_sim.compute()

        # obtain output:

        result = fis_sim.output['Suitability']
        print(f'The crisp suitability output is {round(result,2)}')

    elif mode == 'Control Surface':

        # range of load history:
        xlh = np.linspace(0,10,50)
        # range of weighted travel distance:
        ywtd = np.linspace(0,50,50)
    
        x,y = np.meshgrid(xlh,ywtd)
        z = np.zeros_like(x)

        for i in range(50):
            for j in range(50):
                fis_sim.input['Load History'] = x[i,j]
                fis_sim.input['Weighted Travel Distance'] = y[i,j]
                fis_sim.input['Capabilities'] = capability
                fis_sim.compute()
                z[i,j] = fis_sim.output['Suitability']
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis',
                            linewidth=0.4, antialiased=True)

        ax.view_init(30, 200)
        plt.show()
    else:
        print('Please select one of the two modes')

######################        Main         ######################

load = 2
travel = 14
capability = 1
mode = "Control Surface"

fis(load ,travel, capability, mode)