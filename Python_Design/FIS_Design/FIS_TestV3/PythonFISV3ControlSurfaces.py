"""

This program is a Python implementation of a Fuzzy Inference System (FIS)
that was first designed within MATLAB, using the Fuzzy Logic Designer.

This file serves to examine the surface of control for this FIS.

This particular FIS utilizes the load history, weighted travel distance to task,
and the distance history to make decisions about the suitability of a given robot.

"""
######################## Import Packages ########################

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PythonFISFunctionV3 import *
from matplotlib import cm

#######################       Main       ########################

# define rulebase:
rulebase = fis_create()

# create FIS system from rulebase:
system = ctrl.ControlSystem(rulebase)

# create a simulation instance from the FIS system:
resolution = 32
sim = ctrl.ControlSystemSimulation(system, flush_after_run = resolution * resolution + 1)

# define universes of discourse:
x = np.linspace(0, 10, resolution)  # load history
y = np.linspace(0, 25, resolution)  # distance to task
z = np.linspace(0, 50, resolution)  # total distance travelled

##########   Control Surface 1: LH vs. DTT vs. Suit    ##########
# for this control surface, the values of LH and DTT are varied while the third variable, 
# the total distance travelled, is held constant at half its universe of discourse:

# create meshgrid for 2D surface:
lh, dtt = np.meshgrid(x, y)
u1 = np.zeros_like(lh)

# fix one variable at half its range:
fixed_tdt = max(z) / 2

# loop through the 2D input space and compute suitability for each combination:
counter = 1
for i in range(resolution):
    for j in range(resolution):
        sim.input['Load History'] = lh[i, j]
        sim.input['Distance to Task'] = dtt[i, j]
        sim.input['Total Distance Travelled'] = fixed_tdt
        sim.compute()
        u1[i,j] = sim.output['Suitability']
        print(f'simulation 1 | checking combination: {counter}/{resolution*resolution}', end = '\r')
        counter += 1
print('\n')

# plot the result in 3D:
fig1 = plt.figure(figsize = (8,8))
ax1 = fig1.add_subplot(111, projection = '3d')
ax1.plot_surface(lh, dtt, u1, cmap = 'plasma', linewidth = 0.4, antialiased = True)
ax1.set_title(f'Control Surface 1: LH vs. DTT vs. Suit | TDT: {fixed_tdt}')
ax1.set_xlabel('Load History')
ax1.set_ylabel('Distance to Task')
ax1.set_zlabel('Suitability')


##########   Control Surface 2: LH vs. TDT vs. Suit    ##########
# for this control surface, the values of LH and TDT are varied while the third variable, 
# the distance to task, is held constant at half its universe of discourse:

# create meshgrid for 2D surface:
lh, tdt = np.meshgrid(x, z)
u2 = np.zeros_like(lh)

# fix one variable at half its range:
fixed_dtt = max(y) / 2

# loop through the 2D input space and compute suitability for each combination:
counter = 1
for i in range(resolution):
    for j in range(resolution):
        sim.input['Load History'] = lh[i, j]
        sim.input['Distance to Task'] = fixed_dtt
        sim.input['Total Distance Travelled'] = tdt[i, j]
        sim.compute()
        u2[i,j] = sim.output['Suitability']
        print(f'simulation 2 | checking combination: {counter}/{resolution*resolution}', end = '\r')
        counter += 1
print('\n')

# plot the result in 3D:
fig2 = plt.figure(figsize = (8,8))
ax2 = fig2.add_subplot(111, projection = '3d')
ax2.plot_surface(lh, tdt, u2, cmap = 'plasma', linewidth = 0.4, antialiased = True)
ax2.set_title(f'Control Surface 2: LH vs. TDT vs. Suit | DTT: {fixed_dtt}')
ax2.set_xlabel('Load History')
ax2.set_ylabel('Total Distance Travelled')
ax2.set_zlabel('Suitability')

##########   Control Surface 3: DTT vs. TDT vs. Suit    ##########
# for this control surface, the values of DTT and TDT are varied while the third variable, 
# the load history, is held constant at half its universe of discourse:

# create meshgrid for 2D surface:
dtt, tdt = np.meshgrid(y, z)
u3 = np.zeros_like(dtt)

# fix one variable at half its range:
fixed_lh = max(x) / 2

# loop through the 2D input space and compute suitability for each combination:
counter = 1
for i in range(resolution):
    for j in range(resolution):
        sim.input['Load History'] = fixed_lh
        sim.input['Distance to Task'] = dtt[i, j]
        sim.input['Total Distance Travelled'] = tdt[i, j]
        sim.compute()
        u3[i,j] = sim.output['Suitability']
        print(f'simulation 3 | checking combination: {counter}/{resolution*resolution}', end = '\r')
        counter += 1
print('\n')

# plot the result in 3D:
fig3 = plt.figure(figsize = (8,8))
ax3 = fig3.add_subplot(111, projection = '3d')
ax3.plot_surface(dtt, tdt, u3, cmap = 'plasma', linewidth = 0.4, antialiased = True)
ax3.set_title(f'Control Surface 3: DTT vs. TDT vs. Suit | LH: {fixed_lh}')
ax3.set_xlabel('Distance to Task')
ax3.set_ylabel('Total Distance Travelled')
ax3.set_zlabel('Suitability')

# show plots:
plt.show()



