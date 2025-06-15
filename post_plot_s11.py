import os
import pandas as pd
import matplotlib.pyplot as plt
from settings import GOAL, GFMIN, GFMAX
import numpy as np

GOAL = -6

data_dir = "./data/comparison/"
# files = [f for f in os.listdir(data_dir) if f.startswith("initial") and f.endswith(".csv")]
# files_cst = [f for f in os.listdir(data_dir) if f.startswith("CST") and f.endswith(".csv")]
# files_FNN = [f for f in os.listdir(data_dir) if f.startswith("FNN") and f.endswith(".csv")]
# files_RNN = [f for f in os.listdir(data_dir) if f.startswith("RNN") and f.endswith(".csv")]
# files_TAB = [f for f in os.listdir(data_dir) if f.startswith("TAB") and f.endswith(".csv")]
# files_FNN_weighted = [f for f in os.listdir(data_dir) if f.startswith("weighted") and f.endswith(".csv")]

def parse(case):
    plt.figure(figsize=(10, 5)
    df = pd.read_csv(f"data/comparison/intial_{case}")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    plt.plot(frequency, s11, label='Initial', linestyle=':', color = '#000')
    df = pd.read_csv(f"data/comparison/CST_{case}")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    plt.plot(frequency, s11, label='CST Optimizer', linestyle='--', color = '#4D0')
    df = pd.read_csv(f"data/comparison/FNN_{case}")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    plt.plot(frequency, s11, label='Our Work (FNN)', linestyle='-', color = '#00F')
    df = pd.read_csv(f"data/comparison/weighted_{case}")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    plt.plot(frequency, s11, label='Our Work (weighted FNN)', linestyle='-.', color = '#808')
    df = pd.read_csv(f"data/comparison/RNN_{case}")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    plt.plot(frequency, s11, label='Our Work (RNN)', linestyle='-.', color = '#E00')
    df = pd.read_csv(f"data/comparison/TAB_{case}")  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    plt.plot(frequency, s11, label='Our Work (TabTransformer)', linestyle='-.', color = '#FA4')
    # for file in files[case:case+ran]:
    #     file_path = os.path.join(data_dir, file)
    #     df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    #     frequency = df.iloc[:, 0]
    #     s11 = df.iloc[:, 1]
    #     plt.plot(frequency, s11, label='Initial', linestyle=':', color = '#000')
        
    # for file in files_cst[case:case+ran]:
    #     file_path = os.path.join(data_dir, file)
    #     df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    #     frequency = df.iloc[:, 0]
    #     s11 = df.iloc[:, 1]
    #     plt.plot(frequency, s11, label='CST Optimizer', linestyle='--', color = '#4D0')
        
    # for file in files_FNN[case:case+ran]:
    #     file_path = os.path.join(data_dir, file)
    #     df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    #     frequency = df.iloc[:, 0]
    #     s11 = df.iloc[:, 1]
    #     plt.plot(frequency, s11, label='Our Work (FNN)', linestyle='-', color = '#00F')

    # for file in files_FNN_weighted[case:case+ran]:
    #     file_path = os.path.join(data_dir, file)
    #     df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    #     frequency = df.iloc[:, 0]
    #     s11 = df.iloc[:, 1]
    #     plt.plot(frequency, s11, label='Our Work (weighted FNN)', linestyle='-.', color = '#808')
        
    # for file in files_RNN[case:case+ran]:
    #     file_path = os.path.join(data_dir, file)
    #     df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    #     frequency = df.iloc[:, 0]
    #     s11 = df.iloc[:, 1]
    #     plt.plot(frequency, s11, label='Our Work (RNN)', linestyle='-.', color = '#E00')
        
    # for file in files_TAB[case:case+ran]:
    #     file_path = os.path.join(data_dir, file)
    #     df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    #     frequency = df.iloc[:, 0]
    #     s11 = df.iloc[:, 1]
    #     plt.plot(frequency, s11, label='Our Work (TabTransformer)', linestyle='-.', color = '#FA4')
        
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
    plt.legend()
    plt.grid()
    plt.savefig(f'data/comparison/comparison{case}.png')

for case in range(10): parse(case)
