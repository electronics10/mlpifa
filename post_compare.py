import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
import time
from pifa_controller import MLPIFA
from pifa_controller import INIT_P1, INIT_P2, INIT_P3

if __name__ == "__main__":
    os.makedirs("data/s11", exist_ok=True)
    mlpifa = MLPIFA("antennas/mlpifa.cst") # Call mlpifa file
    # mlpifa.set_environment() # set PIFA and blocks for the first call
    mlpifa.set_frequency_solver()

    # Read predicted data by model # post
    df = pd.read_csv("data/post_prediction.csv", header=None) # post
    predicted_data = df.iloc[:].values.astype(np.float32) # post

for index in range(len(predicted_data)): # Loop through all n samples and run CST optimizer
        mlpifa.delete_results() # clear legacy
        input_seq = list(predicted_data[index, :10])
        print(f"\nSample{index} optimizing...")
        # Update feedx and blocks material
        mlpifa.create_parameters('fx', input_seq[0]) # update fx
        mlpifa.update_distribution(input_seq[1:]) # ignore fx in the first index and update blocks material

        # Without optimization
        # Update parameters # post
        mlpifa.parameters = {"pin_dis":INIT_P1, "patch_len":INIT_P2, "pin_width":INIT_P3} # post
        for key, value in mlpifa.parameters.items(): # post
             print(f"key: {key}; value: {value}") # post
             mlpifa.create_parameters(key, value) # post
        # Without optimization # post
        mlpifa.set_port() # post
        mlpifa.start_simulate() # post
        print(f"Sample{index} solved without optimization") #post
        # Obtain S11 for further inspection # post
        s11 = mlpifa.read('1D Results\\S-Parameters\\S1,1') # [freq, s11 50+j,...]
        s11 = np.abs(np.array(s11))
        s11[:,1] = 20*np.log10(s11[:,1]) # change s11 to dB
        data = pd.DataFrame(s11[:,:-1], columns=['freq', 's11']) # create a DataFrame
        data.to_csv(f'data/s11/s{index}.csv', index=False) # save to CSV
        print(f"S11 saved to 'data/s11/s{index}.csv'")
        # Store data into data.csv for training usage # post
        data = input_seq
        for val in mlpifa.parameters.values(): data.append(val)
        with open('data/data_wo.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)
            print("Input and ouput parameters stored in data/data_wo.csv")
        # Clear legacy for optimization # post
        mlpifa.delete_results() # post
        mlpifa.delete_port() # post

        # Optimized by ML model
        # Update parameters # post
        pre_para = list(predicted_data[index, 10:13]) # post
        mlpifa.parameters = {"pin_dis":pre_para[0], \
                             "patch_len":pre_para[1], \
                                "pin_width":pre_para[2]} # post
        for key, value in mlpifa.parameters.items(): # post
             print(f"key: {key}; value: {value}") # post
             mlpifa.create_parameters(key, value) # post
        # Without optimization # post
        mlpifa.set_port() # post
        mlpifa.start_simulate() # post
        print(f"Sample{index} solved with ML optimization") #post
        # Obtain S11 for further inspection # post
        s11 = mlpifa.read('1D Results\\S-Parameters\\S1,1') # [freq, s11 50+j,...]
        s11 = np.abs(np.array(s11))
        s11[:,1] = 20*np.log10(s11[:,1]) # change s11 to dB
        data = pd.DataFrame(s11[:,:-1], columns=['freq', 's11']) # create a DataFrame
        data.to_csv(f'data/s11/m{index}.csv', index=False) # save to CSV
        print(f"S11 saved to 'data/s11/m{index}.csv'")
        # Store data into data.csv for training usage # post
        data1 = input_seq
        for val in mlpifa.parameters.values(): data1.append(val)
        with open('data/data_ml.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data1)
            print("Input and ouput parameters stored in data/data_ml.csv")
        # Clear legacy for optimization # post
        mlpifa.delete_results() # post
        mlpifa.delete_port() # post

        # Optimized by CST
        mlpifa.set_port()
        start_time = time.time()
        mlpifa.parameters = {"pin_dis":INIT_P1, "patch_len":INIT_P2, "pin_width":INIT_P3} # altered to make optimizer start with these values
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
        data.to_csv(f'data/s11/o{index}.csv', index=False) # save to CSV
        print(f"S11 saved to 'data/s11/o{index}.csv'")
        # Clear legacy for next iteration
        mlpifa.delete_results()
        mlpifa.delete_port() # feed postion may change in next iterationeed postion may change in next iteration
