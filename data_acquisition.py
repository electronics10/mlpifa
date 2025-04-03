import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import time
from pifa_controller import MLPIFA

N_SAMPLES = 2000

if __name__ == "__main__":
    mlpifa = MLPIFA("antennas/mlpifa9.cst") # Call mlpifa file
    # mlpifa.set_environment() # set PIFA and blocks for the first call
    mlpifa.set_frequency_solver()
    input_data = mlpifa.generate_input(N_SAMPLES) # generate input.csv randomly, [[feedx, binary seq],...]
    iter_start = 269 # start data acquisition from sample 'iter_start'

for index in range(iter_start, N_SAMPLES): # Loop through all n samples and run CST optimizer
        mlpifa.delete_results() # clear legacy
        input_seq = list(input_data[index])
        print(f"\nSample{index} optimizing...")
        # Update feedx and blocks material
        command = ['Sub Main']
        command += mlpifa.create_parameters('fx', input_seq[0]) # update fx
        command.append('End Sub')
        mlpifa.excute_vba(command)
        mlpifa.update_distribution(input_seq[1:]) # ignore fx in the first index and update blocks material
        # Optimize
        mlpifa.set_port()
        start_time = time.time()
        mlpifa.optimize() # run CST optimizer
        end_time = time.time()
        print("optimization time =", end_time-start_time)
        print(f"Sample{index} optimized")
        # Store data into data.csv for training usage
        mlpifa.update_parameter_dict()
        data = input_seq
        for val in mlpifa.parameters.values(): data.append(val)
        with open('data/data.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
            print("Input and ouput parameters stored in data/data.csv")
        # Obtain optimized S11 for further inspection
        s11 = mlpifa.read('1D Results\\S-Parameters\\S1,1') # [freq, s11 50+j,...]
        s11 = np.abs(np.array(s11))
        s11[:,1] = 20*np.log10(s11[:,1]) # change s11 to dB
        data = pd.DataFrame(s11[:,:-1], columns=['freq', 's11']) # create a DataFrame
        data.to_csv(f'data/s11/s11_{index}.csv', index=False) # save to CSV
        print(f"S11 saved to 'data/s11/s11_{index}.csv'")
        # Clear legacy for next iteration
        mlpifa.delete_results()
        mlpifa.delete_port() # feed postion may change in next iterationeed postion may change in next iteration