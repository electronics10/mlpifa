import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from models import AntennaMLP
import os
import pickle
from settings import FEEDX_MAX, FEEDX_MIN, BLOCKS_NUM, OUTPUT_LABELS

FOLDER = "artifacts_FNN"
os.makedirs(FOLDER, exist_ok=True)

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# # Load dataset
# data = pd.read_csv('data/data.csv').values
# X = data[:, :BLOCKS_NUM+1]
# y = data[:, BLOCKS_NUM+1:]

# # Normalize inputs and outputs separately
# x_scaler = MinMaxScaler()
# y_scaler = MinMaxScaler()

# X = x_scaler.fit_transform(X)
# y = y_scaler.fit_transform(y)

# # Save scalers
# pickle.dump(x_scaler, open(f"{FOLDER}/x_scaler.pkl", "wb"))
# pickle.dump(y_scaler, open(f"{FOLDER}/y_scaler.pkl", "wb"))

# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
# y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
# X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
# y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Model, Loss, Optimizer
model = AntennaMLP().to(device)

# # Define per-output weights (e.g., [D, L, W])
# output_weights = torch.tensor([1.0, 3.0, 1.0, 1.0]).to(device)  # Adjustable

# class WeightedMSELoss(nn.Module):
#     def __init__(self, weights):
#         super(WeightedMSELoss, self).__init__()
#         self.weights = weights

#     def forward(self, preds, targets):
#         loss = self.weights * (preds - targets) ** 2
#         return loss.mean()

# criterion = WeightedMSELoss(output_weights)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# # Early stopping settings
# early_stopping_patience = 70
# best_loss = float('inf')
# patience_counter = 0

# # Training loop
# num_epochs = 10000
# loss_list = []

# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)
#     loss.backward()
#     optimizer.step()

#     model.eval()
#     with torch.no_grad():
#         test_outputs = model(X_test)
#         test_loss = criterion(test_outputs, y_test)
#         loss_list.append([epoch, loss.item(), test_loss.item()])

#     if epoch % 10 == 0:
#         print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

#     # Early stopping
#     if test_loss.item() < best_loss:
#         best_loss = test_loss.item()
#         best_model_state = model.state_dict()
#         patience_counter = 0
#     else:
#         patience_counter += 1

#     if patience_counter >= early_stopping_patience:
#         print(f"Early stopping at epoch {epoch}")
#         break


# # Save the model
# torch.save(model.state_dict(), f'{FOLDER}/model.pth')

# # === Save loss ===
# loss_list = np.array(loss_list)
# plt.figure(figsize=(6, 5))
# plt.plot(loss_list[:,0:1], loss_list[:,1:2], label='training loss')
# plt.plot(loss_list[:,0:1], loss_list[:,2:3], label='validation loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid()
# plt.title('Training and Validation Loss')
# plt.savefig(f'{FOLDER}/loss.png')
# plt.show()


# # Evaluation and plotting
# model.load_state_dict(torch.load(f"{FOLDER}/model.pth")) # Load best model saved
# model.eval()
# with torch.no_grad():
#     preds = model(X_test)
#     preds = preds.cpu().numpy()
#     y_true = y_test.cpu().numpy()

# # Inverse transform
# preds = y_scaler.inverse_transform(preds)
# y_true = y_scaler.inverse_transform(y_true)

# # === Plot each output ===
# for i in range(len(OUTPUT_LABELS)):
#     plt.figure(figsize=(6, 5))
#     plt.scatter(y_true[:, i], preds[:, i], alpha=0.6)
#     plt.plot([y_true[:, i].min(), y_true[:, i].max()],
#              [y_true[:, i].min(), y_true[:, i].max()], 'r--')
#     plt.xlabel('True Value')
#     plt.ylabel('Predicted Value')
#     plt.title(f'{OUTPUT_LABELS[i]}: Predicted vs True')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(f'{FOLDER}/output_{i+1}_plot.png')
#     plt.show()
#     plt.close()

# # === Compute and save test loss ===
# mse = np.mean((preds - y_true) ** 2)
# with open(f"{FOLDER}/test_loss.txt", "w") as f:
#     f.write(f"MSE on test set: {mse:.6f}\n")

# print(f"Test MSE: {mse:.6f}")


# === Output 5 predictions ===
def generate_input():
    np.random.seed(30)
    samples = 15
    fx = np.random.rand(samples, 1).astype(np.float32) # generate feed x position
    fx = fx*(FEEDX_MAX - FEEDX_MIN) + FEEDX_MIN
    np.random.seed(30)
    blocks = np.random.randint(0, 2, (samples, BLOCKS_NUM))
    data = np.concatenate((fx, blocks), axis=1).astype(np.float32)
    return data

X = np.array([[5.465447373,1,1,1,1,1,1,1,1,0,1],[6.056937754,1,1,1,1,0,1,0,1,0,1]]) # generate_input()
x_scaler = pickle.load(open(f"{FOLDER}/x_scaler.pkl", "rb"))
y_scaler = pickle.load(open(f"{FOLDER}/y_scaler.pkl", "rb"))
X_scaled = x_scaler.transform(X)
X_test = torch.tensor(X_scaled, dtype=torch.float32).to(device)

model.load_state_dict(torch.load(f"{FOLDER}/model.pth")) # Load best model saved
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test).cpu().numpy()
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

# df = pd.DataFrame(np.concatenate((X, y_pred), axis=1))
# df.to_csv(f'{FOLDER}/post_prediction.csv', header = None, index=False)
print(np.concatenate((X, y_pred), axis=1))
