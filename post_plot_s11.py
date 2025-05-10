import os
import pandas as pd
import matplotlib.pyplot as plt
from settings import GOAL, GFMIN, GFMAX

GOAL = -6

data_dir = "./data/comparison/"
files = [f for f in os.listdir(data_dir) if f.startswith("i") and f.endswith(".csv")]
files_cst = [f for f in os.listdir(data_dir) if f.startswith("o") and f.endswith(".csv")]
files_ml = [f for f in os.listdir(data_dir) if f.startswith("m") and f.endswith(".csv")]
files_ml_weighted = [f for f in os.listdir(data_dir) if f.startswith("w") and f.endswith(".csv")]

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
        plt.plot(frequency, s11, label='CST', linestyle='--', color = '#4D0')
    for file in files_ml[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        plt.plot(frequency, s11, label='Our Work', linestyle='-', color = '#00F')
    for file in files_ml_weighted[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        plt.plot(frequency, s11, label='Our Work (weighted)', linestyle='-', color = '#F00')
    plt.axvline(x = GFMIN, linestyle='-.', color = '#999')
    plt.axvline(x = GFMAX, linestyle='-.', color = '#999')
    plt.axhline(y = GOAL, linestyle='-.', color = '#999')
    plt.xlabel("Frequency")
    plt.ylabel("S11")
    plt.title(f"Comparison{case}")
    plt.legend()
    plt.grid()
    plt.savefig(f'data/comparison/comparison{case}.png')

for case in range(len(files)): parse(case, 1)
