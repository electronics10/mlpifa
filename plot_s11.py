import os
import pandas as pd
import matplotlib.pyplot as plt

data_dir = "./data/s11/"
files = [f for f in os.listdir(data_dir) if f.startswith("s11_") and f.endswith(".csv")]

plt.figure(figsize=(10, 5))

for file in files:
    file_path = os.path.join(data_dir, file)
    df = pd.read_csv(file_path)  # Assuming headers are present in CSV
    frequency = df.iloc[:, 0]
    s11 = df.iloc[:, 1]
    label = file.replace(".csv", "")
    plt.plot(frequency, s11, label=label)

plt.xlabel("Frequency")
plt.ylabel("S11")
plt.title("S11 vs Frequency")
plt.legend()
plt.grid()
plt.show()

