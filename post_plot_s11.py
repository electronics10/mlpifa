import os
import pandas as pd
import matplotlib.pyplot as plt
from settings import GOAL, GFMIN, GFMAX

GOAL = -6

data_dir = "./data/comparison/"
files = [f for f in os.listdir(data_dir) if f.startswith("initial") and f.endswith(".csv")]
files_cst = [f for f in os.listdir(data_dir) if f.startswith("CST") and f.endswith(".csv")]
files_FNN = [f for f in os.listdir(data_dir) if f.startswith("FNN") and f.endswith(".csv")]
files_RNN = [f for f in os.listdir(data_dir) if f.startswith("RNN") and f.endswith(".csv")]
files_TAB = [f for f in os.listdir(data_dir) if f.startswith("TAB") and f.endswith(".csv")]
files_FNN_weighted = [f for f in os.listdir(data_dir) if f.startswith("weighted") and f.endswith(".csv")]

def parse(case, ran):
    plt.figure(figsize=(10, 5))
    for file in files[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        plt.plot(frequency, s11, label='Initial', linestyle=':', color = '#000')
        
    for file in files_cst[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        plt.plot(frequency, s11, label='CST Optimizer', linestyle='--', color = '#4D0')
        
    for file in files_FNN[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        plt.plot(frequency, s11, label='Our Work (FNN)', linestyle='-', color = '#00F')

    for file in files_FNN_weighted[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        plt.plot(frequency, s11, label='Our Work (weighted FNN)', linestyle='-.', color = '#808')
        
    for file in files_RNN[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        plt.plot(frequency, s11, label='Our Work (RNN)', linestyle='-.', color = '#E00')
        
    for file in files_TAB[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        plt.plot(frequency, s11, label='Our Work (TabTransformer)', linestyle='-.', color = '#FA4')
        
    # plt.axvline(x = GFMIN, linestyle='-.', color = '#999')
    # plt.axvline(x = GFMAX, linestyle='-.', color = '#999')
    # plt.axhline(y = GOAL, linestyle='-.', color = '#999')
    plt.xlim(2, 3)
    plt.ylim(-20, 0)
    plt.xticks(np.arange(2, 3, 0.1))
    plt.xlabel("Frequency")
    plt.ylabel("S11")
    plt.title(f"Case{case}")
    plt.legend()
    plt.grid()
    plt.savefig(f'data/comparison/comparison{case}.png')

for case in range(len(files)): parse(case, 1)
