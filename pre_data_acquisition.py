import numpy as np
import csv
import pandas as pd
import time
from pifa_controller import MLPIFA
from pifa_controller import INIT_P1, INIT_P2, INIT_P3
import os


if __name__ == "__main__":
    os.makedirs("./antennas", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./data/s11", exist_ok=True)

    mlpifa = MLPIFA("antennas/mlpifa.cst") # Call mlpifa file
    # mlpifa.set_environment() # set PIFA and blocks for the first call
    mlpifa.set_frequency_solver()
    input_data = mlpifa.generate_input() # generate input.csv randomly, [[feedx, binary seq],...]
    iter_start = 0 # start data acquisition from sample 'iter_start'

for index in range(iter_start, len(input_data)): # Loop through all n samples and run CST optimizer
        mlpifa.delete_results() # clear legacy
        input_seq = list(input_data[index])
        print(f"\nSample{index} optimizing...")
        # Update feedx and blocks material
        feedx = input_seq[0]
        mlpifa.create_parameters('fx', feedx) # update fx
        mlpifa.update_distribution(input_seq[1:]) # ignore fx in the first index and update blocks material

        # Optimize
        mlpifa.set_port()
        start_time = time.time()
        mlpifa.parameters = {"pin_dis":INIT_P1, "patch_len":INIT_P2, "pin_width":INIT_P3} # altered to make optimizer start with these values
        mlpifa.optimize(feedx) # run CST optimizer
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
        data.to_csv(f'data/s11/o{index}.csv', index=False) # save to CSV
        print(f"S11 saved to 'data/s11/o{index}.csv'")
        # Clear legacy for next iteration
        mlpifa.delete_results()
        mlpifa.delete_port() # feed postion may change in next iterationeed postion may change in next iteration
