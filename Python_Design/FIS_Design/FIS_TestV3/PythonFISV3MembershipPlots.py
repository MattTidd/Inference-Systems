"""

This program is a Python implementation of a Fuzzy Inference System (FIS)
that was first designed within MATLAB, using the Fuzzy Logic Designer.

This file serves to examine the membership functions for the FIS.

This particular FIS utilizes the load history, weighted travel distance to task,
and the distance history to make decisions about the suitability of a given robot.

"""
######################## Import Packages ########################

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt


#######################       Main       ########################

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

lh_lo = fuzz.trimf(lh.universe, [0, 0, 6])
lh_md = fuzz.trimf(lh.universe, [5/6, 5, 55/6])
lh_hi = fuzz.trimf(lh.universe, [4, 10, 10])

dtt_lo = fuzz.trimf(dtt.universe, [0, 0, 15])
dtt_md = fuzz.trimf(dtt.universe, [25/12, 12.5, 275/12])
dtt_hi = fuzz.trimf(dtt.universe, [10, 25, 25])

tdt_lo = fuzz.trimf(tdt.universe, [0, 0, 30])
tdt_md = fuzz.trimf(tdt.universe, [25/6, 25, 275/6])
tdt_hi = fuzz.trimf(tdt.universe, [15, 50, 50])

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
suit_vl = fuzz.trimf(suit.universe, [0, 0, 25/12])
suit_lo = fuzz.trimf(suit.universe, [5/12, 2.5, 55/12])
suit_md = fuzz.trimf(suit.universe, [35/12, 5, 85/12])
suit_hi = fuzz.trimf(suit.universe, [65/12, 7.5, 115/12])
suit_vhi = fuzz.trimf(suit.universe, [95/12, 10, 10])

# Visualize these universes and membership functions
fig1, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(3.5, 10))
fig1.tight_layout()

# for load history:
ax0.plot(lh_range, lh_lo, 'b', linewidth = 1.5, label = 'Low')
ax0.plot(lh_range, lh_md, 'g', linewidth = 1.5, label = 'Medium')
ax0.plot(lh_range, lh_hi, 'r', linewidth = 1.5, label = 'High')
ax0.set_title('Load History')
ax0.minorticks_on()
ax0.grid(which = 'major', color = 'black', linestyle = '--', linewidth = 0.5)
ax0.grid(which = 'minor', color = 'gray', linestyle = ':', linewidth = 0.25)

# for distance to task:
ax1.plot(dtt_range, dtt_lo, 'b', linewidth = 1.5, label = 'Low')
ax1.plot(dtt_range, dtt_md, 'g', linewidth = 1.5, label = 'Medium')
ax1.plot(dtt_range, dtt_hi, 'r', linewidth = 1.5, label = 'High')
ax1.set_title('Distance to Task')
ax1.minorticks_on()
ax1.grid(which = 'major', color = 'black', linestyle = '--', linewidth = 0.5)
ax1.grid(which = 'minor', color = 'gray', linestyle = ':', linewidth = 0.25)

# for total distance travelled:
ax2.plot(tdt_range, tdt_lo, 'b', linewidth = 1.5, label = 'L')
ax2.plot(tdt_range, tdt_md, 'g', linewidth = 1.5, label = 'M')
ax2.plot(tdt_range, tdt_hi, 'r', linewidth = 1.5, label = 'H')
ax2.set_title('Total Distance Travelled')
ax2.minorticks_on()
ax2.grid(which = 'major', color = 'black', linestyle = '--', linewidth = 0.5)
ax2.grid(which = 'minor', color = 'gray', linestyle = ':', linewidth = 0.25)
ax2.legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.1), fancybox = True, ncol = 3)

# for the output variable:

fig = plt.figure(figsize = (3.5, 2.5))

plt.plot(suit_range, suit_vl, 'c', linewidth = 1.5, label = 'VL')
plt.plot(suit_range, suit_lo, 'b', linewidth = 1.5, label = 'L')
plt.plot(suit_range, suit_md, 'g', linewidth = 1.5, label = 'M')
plt.plot(suit_range, suit_hi, 'r', linewidth = 1.5, label = 'H')
plt.plot(suit_range, suit_vhi, 'm', linewidth = 1.5, label = 'VH')
plt.title('Suitability')
plt.minorticks_on()
plt.grid(which = 'major', color = 'black', linestyle = '--', linewidth = 0.5)
plt.grid(which = 'minor', color = 'gray', linestyle = ':', linewidth = 0.25)
plt.legend(loc='upper center', bbox_to_anchor =(0.5, -0.1), fancybox = True, ncol = 5)

fig1.savefig('input_membership.png', dpi = 1000, bbox_inches = 'tight')
plt.savefig("output_membership.png", dpi = 1000, bbox_inches='tight')
plt.show()
