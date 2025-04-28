import torch
import torch.nn as nn
import pickle
import pandas as pd
import numpy as np
from models import AntennaMLP

SEED = 30
FEEDX_MAX = 12
FEEDX_MIN = 8
BLOCKS_NUM = 9

SAMPLES = 5

# === Load test input data ===
def generate_input():
    np.random.seed(SEED)
    fx = np.random.rand(SAMPLES, 1).astype(np.float32) # generate feed x position
    fx = fx*(FEEDX_MAX - FEEDX_MIN) + FEEDX_MIN
    np.random.seed(SEED)
    blocks = np.random.randint(0, 2, (SAMPLES, BLOCKS_NUM))
    data = np.concatenate((fx, blocks), axis=1).astype(np.float32)
    return data

X = generate_input()

# === Load scalers ===
x_scaler = pickle.load(open("artifacts/x_scaler.pkl", "rb"))
y_scaler = pickle.load(open("artifacts/y_scaler.pkl", "rb"))

X_scaled = x_scaler.transform(X)
X_test = torch.tensor(X_scaled)

model = AntennaMLP()
model.load_state_dict(torch.load("artifacts/mlp_model.pt"))
model.eval()

# === Predict ===
with torch.no_grad():
    y_pred_scaled = model(X_test).numpy()
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

df = pd.DataFrame(np.concatenate((X, y_pred), axis=1))
df.to_csv('data/post_prediction.csv', header = None, index=False)
