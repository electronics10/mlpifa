import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tab_transformer_pytorch import TabTransformer
from torch.utils.data import DataLoader, TensorDataset
import os
import pickle
from settings import FEEDX_MAX, FEEDX_MIN, BLOCKS_NUM, OUTPUT_LABELS

FOLDER = "artifacts_TAB"
os.makedirs(FOLDER, exist_ok=True)

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_built() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Load dataset
data = pd.read_csv('data/data.csv').values
X = data[:, :BLOCKS_NUM+1]
y = data[:, BLOCKS_NUM+1:]

# Normalize inputs and outputs separately
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

X = x_scaler.fit_transform(X)
y = y_scaler.fit_transform(y)

# Save scalers
pickle.dump(x_scaler, open(f"{FOLDER}/x_scaler.pkl", "wb"))
pickle.dump(y_scaler, open(f"{FOLDER}/y_scaler.pkl", "wb"))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Prepare data for TabTransformer
# Convert to torch tensors and move to device
y_train_tensor = torch.FloatTensor(y_train).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)
categorical_columns = list(range(1, BLOCKS_NUM+1))
continuous_columns = [0]
X_train_cat = X_train[:, categorical_columns].astype(int)
X_test_cat = X_test[:, categorical_columns].astype(int)
X_train_cont = X_train[:, continuous_columns]
X_test_cont = X_test[:, continuous_columns]
X_train_cat_tensor = torch.LongTensor(X_train_cat).to(device)
X_test_cat_tensor = torch.LongTensor(X_test_cat).to(device)
X_train_cont_tensor = torch.FloatTensor(X_train_cont).to(device)
X_test_cont_tensor = torch.FloatTensor(X_test_cont).to(device)

# Model
model = TabTransformer(
            categories=[2] * BLOCKS_NUM,
            num_continuous=1,
            dim=64,
            dim_out=len(OUTPUT_LABELS),
            depth=3,
            heads=4,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
        ).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Early stopping settings
early_stopping_patience = 70
best_loss = float('inf')
patience_counter = 0

# Training loop
num_epochs = 1000
loss_list = []

tab_dataset = TensorDataset(X_train_cat_tensor, X_train_cont_tensor, y_train_tensor)
tab_test_dataset = TensorDataset(X_test_cat_tensor, X_test_cont_tensor, y_test_tensor)

train_loader = DataLoader(tab_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=0)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_cat_batch, X_cont_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_cat_batch, X_cont_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_cat_batch)

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_cat_tensor, X_test_cont_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        loss_list.append([epoch, total_loss, test_loss.item()])

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Train Loss: {total_loss:.4f}, Test Loss: {test_loss.item():.4f}")

    # Early stopping
    if test_loss.item() < best_loss:
        best_loss = test_loss.item()
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= early_stopping_patience:
        print(f"Early stopping at epoch {epoch}")
        break


# Save the model
torch.save(model.state_dict(), f'{FOLDER}/model.pth')

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
plt.savefig(f'{FOLDER}/loss.png')
plt.show()


# Evaluation and plotting
model.load_state_dict(torch.load(f"{FOLDER}/model.pth"))
model.eval()
with torch.no_grad():
    preds = model(X_test_cat_tensor, X_test_cont_tensor).cpu().numpy()
    y_true = y_test_tensor.cpu().numpy()

# Inverse transform
preds = y_scaler.inverse_transform(preds)
y_true = y_scaler.inverse_transform(y_true)

# === Plot each output ===
for i in range(len(OUTPUT_LABELS)):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true[:, i], preds[:, i], alpha=0.6)
    plt.plot([y_true[:, i].min(), y_true[:, i].max()],
             [y_true[:, i].min(), y_true[:, i].max()], 'r--')
    plt.xlabel('True Value')
    plt.ylabel('Predicted Value')
    plt.title(f'{OUTPUT_LABELS[i]}: Predicted vs True')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{FOLDER}/output_{i+1}_plot.png')
    plt.show()
    plt.close()

# === Compute and save test loss ===
mse = np.mean((preds - y_true) ** 2)
with open(f"{FOLDER}/test_loss.txt", "w") as f:
    f.write(f"MSE on test set: {mse:.6f}\n")

print(f"Test MSE: {mse:.6f}")

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

X = generate_input()
x_scaler = pickle.load(open(f"{FOLDER}/x_scaler.pkl", "rb"))
y_scaler = pickle.load(open(f"{FOLDER}/y_scaler.pkl", "rb"))
X_test = x_scaler.transform(X)

categorical_columns = list(range(1, BLOCKS_NUM+1))
continuous_columns = [0]
X_test_cat = X_test[:, categorical_columns].astype(int)
X_test_cont = X_test[:, continuous_columns]
X_test_cat_tensor = torch.LongTensor(X_test_cat).to(device)
X_test_cont_tensor = torch.FloatTensor(X_test_cont).to(device)

model.load_state_dict(torch.load(f"{FOLDER}/model.pth"))
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_cat_tensor, X_test_cont_tensor).cpu().numpy()
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

df = pd.DataFrame(np.concatenate((X, y_pred), axis=1))
df.to_csv(f'{FOLDER}/post_prediction.csv', header = None, index=False)
