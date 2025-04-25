import os
import pandas as pd
import matplotlib.pyplot as plt
from pifa_controller import GOAL, GFMIN, GFMAX

data_dir = "./data/s11/"
files = [f for f in os.listdir(data_dir) if f.startswith("s") and f.endswith(".csv")]
files_cst = [f for f in os.listdir(data_dir) if f.startswith("o") and f.endswith(".csv")]
files_ml = [f for f in os.listdir(data_dir) if f.startswith("m") and f.endswith(".csv")]

plt.figure(figsize=(10, 5))

def parse(case, ran):
    for file in files[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        label = file.replace(".csv", "")
        plt.plot(frequency, s11, label=label, linestyle=':', color = '#000')
    for file in files_cst[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        label = file.replace(".csv", "")
        plt.plot(frequency, s11, label=label, linestyle='--', color = '#4D0')
    for file in files_ml[case:case+ran]:
        file_path = os.path.join(data_dir, file)
        df = pd.read_csv(file_path)  # Assuming headers are present in CSV
        frequency = df.iloc[:, 0]
        s11 = df.iloc[:, 1]
        label = file.replace(".csv", "")
        plt.plot(frequency, s11, label=label, linestyle='-', color = '#00F')

for case in range(len(files)): parse(case, 1)

plt.axvline(x = GFMIN, linestyle='-.', color = '#999')
plt.axvline(x = GFMAX, linestyle='-.', color = '#999')
plt.axhline(y = GOAL, linestyle='-.', color = '#999')
plt.xlabel("Frequency")
plt.ylabel("S11")
plt.title("s: Not optimized / o: Optimized (by CST) / m: Our work")
plt.legend()
plt.grid()
plt.savefig(f'comparison{CASE}.png')
plt.show()
