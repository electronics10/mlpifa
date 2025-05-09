from settings import BLOCKS_NUM, REGIONX, REGIONY
import pandas as pd
import numpy as np
import csv

LFREQ = 2.4 # GHz
HFREQ = 2.5 # GHz
GOAL = -6 # dB
BOUND = -1 # dB

df = pd.read_csv("data/data.csv", header=None)
output_file = "data/data_filtered.csv"

for index in range(len(df)):
    # Read s11_{index}
    df_s11 = pd.read_csv(f"data/s11/s11_{index}.csv")
    j = 0
    quality = 0
    while True:
        # Read fequency
        freq = df_s11.iloc[j, 0]
        if freq >= LFREQ:
            # Read s11
            s11 = df_s11.iloc[j, 1]
            # Record quality
            quality += GOAL - s11
        if freq >= HFREQ: break
        j += 1
    normalizer_quality = quality/j
    if normalizer_quality < BOUND: 
        with open('data/eliminated_index.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([index]) 
    else:
        with open(output_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(df.iloc[index, :])
        with open('data/saved_index.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([index])   
