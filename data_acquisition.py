import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import time
# need autotune environment instead of mlpifa for cst communication
from test import Controller


### Read input data for binary material distribution from input.csv
## Load data
csv_input = pd.read_csv('data/input.csv')
input_data = csv_input.iloc[:, :8].values  # 8 binary inputs [[random1], [random2],...]


### Define CST optimization process for each sample
def cst_optimization(binary_sequence):
    binary_sequence = list(binary_sequence)

    ##1 Optimization
    # open PIFA CST file
    mlpifa = Controller("mlpifa.cst")
    # update material distribution
    mlpifa.update_distribution(binary_sequence)
    # update initial PIFA parameters
    pin_pos = 5.0
    patch_len = 2.0
    pin_width = 0.5
    parameters = {pin_pos:5.0, patch_len:2.0, pin_width:0.5}
    for key, value in parameters.items(): mlpifa.create_parameters(key, value)
    # run CST optimizer
    start_time = time.time()
    end_time = time.time()
    print("optimization time =", end_time-start_time)
    # obtain optimized PIFA parameters
    # obtain optimized PIFA S11 for further verification
    ##1 Test example for optimization
    pin_pos = 5.0 + 1.5 * np.sum(binary_sequence) + np.random.rand()*0.5
    patch_len = 2.0 + 0.5 * np.sum(binary_sequence) + np.random.rand()*0.2
    pin_width = 0.5 + 0.2 * np.sum(binary_sequence) + np.random.rand()*0.1

    ##2 Record parameters to data.csv
    output = binary_sequence
    output.append(pin_pos)
    output.append(patch_len)
    output.append(pin_width)
    with open('data/data.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(output)
        print("Input and ouput parameters recorded.")

    return None


### Acquire data samples from CST
## Loop through all n samples and run CST optimizer
for index in range(len(input_data)):
    print(f"Sample{index}")
    cst_optimization(input_data[index])
## Add csv column name (header) by pandas since we use pandas to read in training code
df = pd.read_csv('data/data.csv', header=None)
columns=[f'region{i+1}' for i in range(8)]
columns.append('pin_pos')
columns.append('patch_len')
columns.append('pin_width')
df.columns = columns
df.to_csv('data/data.csv', index=False)




