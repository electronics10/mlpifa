import os
import pandas as pd
import matplotlib.pyplot as plt
from settings import GOAL, GFMIN, GFMAX
import numpy as np

GOAL = -6

def parse(case):
    plt.figure(figsize=(10, 5))
    df = pd.read_csv(f"data/comparison/initial_{case}.csv")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    # plt.plot(frequency, s11, label='Initial', linestyle=':', color = '#000')
    plt.plot(frequency, s11, label='Initial', linestyle=(0, (3,1,1,1,1,1)), color = '#000')
    
    df = pd.read_csv(f"data/comparison/CST_{case}.csv")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    # plt.plot(frequency, s11, label='CST Optimizer', linestyle='--', color = '#4D0')
    plt.plot(frequency, s11, label='CST Optimizer', linestyle=(0, (3,1,1,1)), color = '#00F')
    
    df = pd.read_csv(f"data/comparison/FNN_{case}.csv")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    # plt.plot(frequency, s11, label='Our Work (FNN)*', linestyle='-', color = '#00F')
    plt.plot(frequency, s11, label='Our Work (FNN)*', linestyle='solid', color = '#F00')
    
    df = pd.read_csv(f"data/comparison/weighted_{case}.csv")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    # plt.plot(frequency, s11, label='Our Work (weighted FNN)', linestyle='-.', color = '#808')
    plt.plot(frequency, s11, label='Our Work (weighted FNN)', linestyle=(0, (5, 10)), color = '#F00')
    
    df = pd.read_csv(f"data/comparison/RNN_{case}.csv")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    # plt.plot(frequency, s11, label='Our Work (RNN)', linestyle='-.', color = '#E00')
    plt.plot(frequency, s11, label='Our Work (RNN)', linestyle=(0, (3, 10, 1, 10)), color = '#F00')
    
    df = pd.read_csv(f"data/comparison/TAB_{case}.csv")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    # plt.plot(frequency, s11, label='Our Work (TabTransformer)', linestyle='-.', color = '#FA4')
    plt.plot(frequency, s11, label='Our Work (TabTransformer)', linestyle=(0, (1, 10)), color = '#F00')

        
    # plt.axvline(x = GFMIN, linestyle='-.', color = '#999')
    # plt.axvline(x = GFMAX, linestyle='-.', color = '#999')
    # plt.axhline(y = GOAL, linestyle='-.', color = '#999')
    plt.xlim(2, 3)
    plt.ylim(-20, 0)
    plt.xticks(np.arange(2, 3.1, 0.1))
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("S11 (dB)")
    # Handle
    if case == 0: plt.title(f"Case{case+1}")
    elif case == 1: plt.title(f"Case{case-1}")
    else: plt.title(f"Case{case}")
    plt.legend(labelcolor='linecolor')
    # plt.grid()
    plt.savefig(f'data/comparison/comparison{case}.png')

for case in range(10): parse(case)
