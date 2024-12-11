"""

This file utilizes FISV3 to generate data that will be used to train an 
adaptive neuro-fuzzy inference system (ANFIS).

This process involves generating data randomly within the universe of 
discourse of each variable, and using this generated data to infer the 
suitability of a given robot. This is appended to a dataframe and exported
as a CSV.

"""
######################## Import Packages ########################

import pandas as pd
import numpy as np
from PythonFISFunctionV3 import *

#######################  Generate  Data  ########################
# number of iterations for data generation:
iterations = 10
# max value of universe of discourse for load:      
max_ud_load = 10
# max value of universe of discourse for distance to task:        
max_ud_dtt = 25
# max value of universe of discourse for total distance travelled:    
max_ud_tdt = 50

# instantiate rulebase for FIS:
rulebase = fis_create()

# create dataframe for export:
columns = ['Load History', 'Distance to Task', 'Total Distance Travelled', 'Suitability']
df = pd.DataFrame(np.zeros((iterations, len(columns))), columns = columns)

for i in range(iterations):
    # randomly generate three robot parameters:
    load = np.random.randint(0, max_ud_load + 1)
    distance = np.random.randint(0, max_ud_dtt + 1)
    travelled = np.random.randint(0, max_ud_tdt + 1)

    # calculate suitability:
    suit = fis_solve(rulebase, load, distance, travelled)

    # create next row, append to df:
    next_row = [load, distance, travelled, suit]
    df.iloc[i] = next_row

# export dataframe as a CSV:
df.to_csv('V3_Data.csv', index = False)
