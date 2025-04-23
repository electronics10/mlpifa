import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import platform
import models as mymd

# === Load dataset ===
df = pd.read_csv("data/data.csv")
X = df.iloc[:, :10].values.astype(np.float32)
y = df.iloc[:, 10:13].values.astype(np.float32)

# === Normalize ===
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

# Save scalers
os.makedirs("artifacts", exist_ok=True)
pickle.dump(x_scaler, open("artifacts/x_scaler.pkl", "wb"))
pickle.dump(y_scaler, open("artifacts/y_scaler.pkl", "wb"))

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Convert to torch tensors ===
X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
X_test = torch.tensor(X_test)
y_test = torch.tensor(y_test)

model = mymd.MLP()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_fn = nn.MSELoss()

# === Training loop ===
loss_list = []
for epoch in range(1750):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    # Record training and validation loss
    model.eval()
    val_pred = model(X_test)
    val_loss = loss_fn(val_pred, y_test)
    if epoch % 50 == 0: print(f"Epoch {epoch}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
    loss_list.append([epoch, loss.item(), val_loss.item()])

# === Save loss ===
loss_list = np.array(loss_list)
plt.figure(figsize=(6, 5))
plt.plot(loss_list[:,0:1], loss_list[:,1:2], label='training loss')
plt.plot(loss_list[:,0:1], loss_list[:,2:3], label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.title('Training and Validation Loss')
plt.savefig(f'artifacts/loss.png')

# === Save model ===
torch.save(model.state_dict(), "artifacts/mlp_model.pt")

# === Predict ===
os.makedirs("results", exist_ok=True)
with torch.no_grad():
    y_pred_scaled = model(X_test).numpy()
    y_pred = y_scaler.inverse_transform(y_pred_scaled)
    y_true = y_scaler.inverse_transform(y_test.numpy())

# === Save predictions for review ===
np.savetxt("results/predictions.csv", np.hstack([y_true, y_pred]), delimiter=",", header="true_1,true_2,true_3,pred_1,pred_2,pred_3", comments="")

# === Compute and save test loss ===
mse = np.mean((y_pred - y_true) ** 2)
with open("results/test_loss.txt", "w") as f:
    f.write(f"MSE on test set: {mse:.6f}\n")

print(f"Test MSE: {mse:.6f}")

# === Plot each output ===
output_labels = ['D', 'L', 'W'] # pin_dis, patch_len, pin_width
for i in range(3):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
    plt.plot([y_true[:, i].min(), y_true[:, i].max()],
             [y_true[:, i].min(), y_true[:, i].max()], 'r--')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f'{output_labels[i]}: Predicted vs True')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/output_{i+1}_plot.png')
    plt.close()
