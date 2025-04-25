import os
import pandas as pd
import matplotlib.pyplot as plt

CASE = 1

data_dir = "./data/s11/"
files = [f for f in os.listdir(data_dir) if f.startswith("s") and f.endswith(".csv")]
files_cst = [f for f in os.listdir(data_dir) if f.startswith("o") and f.endswith(".csv")]
files_ml = [f for f in os.listdir(data_dir) if f.startswith("m") and f.endswith(".csv")]

plt.figure(figsize=(10, 5))

for file in files[CASE:CASE+1]:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    label = file.replace(".csv", "")
    plt.plot(frequency, s11, label=label, linestyle=':', color = '#000')
for file in files_cst[CASE:CASE+1]:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    label = file.replace(".csv", "")
    plt.plot(frequency, s11, label=label, linestyle='--', color = '#4D0')
for file in files_ml[CASE:CASE+1]:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    label = file.replace(".csv", "")
    plt.plot(frequency, s11, label=label, linestyle='-', color = '#00F')

plt.axvline(x = 2.3, linestyle='-.', color = '#999')
plt.axvline(x = 2.6, linestyle='-.', color = '#999')
plt.axhline(y = -10, linestyle='-.', color = '#999')
plt.xlabel("Frequency")
plt.ylabel("S11")
plt.title("s: Not optimized / o: Optimized (by CST) / m: Our work")
plt.legend()
plt.grid()
plt.savefig(f'comparison{CASE}.png')
plt.show()
