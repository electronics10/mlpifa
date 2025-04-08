# load_and_plot.py
import json  # Add this import for JSON handling
import platform
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pickle
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.loader import DataLoader as GeometricDataLoader  # Add for batched inference
import xgboost as xgb
from tab_transformer_pytorch import TabTransformer

# Device selection function
def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device
    elif platform.system() == "Darwin" and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using MPS device (Apple Silicon GPU)")
        return device
    else:
        device = torch.device("cpu")
        print("CUDA and MPS not available, falling back to CPU")
        return device

# Set device
device = get_device()

# Load best hyperparameters
with open('model/best_params.json', 'r') as f:
    best_params = json.load(f)
print("Loaded best hyperparameters:", best_params)

# Load data
data = pd.read_csv('data/data.csv', header=None)
X = data.iloc[:, :10].values
y = data.iloc[:, 10:].values

# Load scalers
with open('model/scaler_X.pkl', 'rb') as f:
    scaler_X = pickle.load(f)
with open('model/scaler_y.pkl', 'rb') as f:
    scaler_y = pickle.load(f)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize
X_test = scaler_X.transform(X_test)
y_test_orig = y_test
y_test = scaler_y.transform(y_test)

# Convert to tensors and move to device
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# 1. XGBoost
with open('model/xgb_model.pkl', 'rb') as f:
    xgb_model = pickle.load(f)
preds_xgb = xgb_model.predict(X_test)
preds_xgb_orig = scaler_y.inverse_transform(preds_xgb)

# 2. FNN Model
class FNN(nn.Module):
    def __init__(self, hidden1, hidden2, dropout_rate):
        super(FNN, self).__init__()
        self.fc1 = nn.Linear(10, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Initialize FNN with best hyperparameters
fnn_params = best_params['fnn']
fnn_model = FNN(
    hidden1=fnn_params['hidden1'],
    hidden2=fnn_params['hidden2'],
    dropout_rate=fnn_params['dropout_rate']
).to(device)
fnn_model.load_state_dict(torch.load('model/fnn_model.pth'))
fnn_model.eval()
with torch.no_grad():
    preds_fnn = fnn_model(X_test_tensor).cpu().numpy()
preds_fnn_orig = scaler_y.inverse_transform(preds_fnn)

# 3. TabTransformer
categorical_columns = list(range(1, 10))
continuous_columns = [0]
X_test_cat = X_test[:, categorical_columns].astype(int)
X_test_cont = X_test[:, continuous_columns]
X_test_cat_tensor = torch.LongTensor(X_test_cat).to(device)
X_test_cont_tensor = torch.FloatTensor(X_test_cont).to(device)

# Initialize TabTransformer with best hyperparameters
tab_params = best_params['tabtransformer']
tab_model = TabTransformer(
    categories=[2] * 9,
    num_continuous=1,
    dim=tab_params['dim'],
    dim_out=3,
    depth=tab_params['depth'],
    heads=tab_params['heads'],
    attn_dropout=0.1,
    ff_dropout=0.1,
    mlp_hidden_mults=(4, 2),
).to(device)
tab_model.load_state_dict(torch.load('model/tab_model.pth'))
tab_model.eval()
with torch.no_grad():
    preds_tab = tab_model(X_test_cat_tensor, X_test_cont_tensor).cpu().numpy()
preds_tab_orig = scaler_y.inverse_transform(preds_tab)

# 4. GCN Model
class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(1, 8)
        self.conv2 = GCNConv(8, 8)
        self.fc1 = nn.Linear(8 * 10, 16)
        self.fc2 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        x = self.relu(self.conv1(x, edge_index, edge_attr))
        x = self.relu(self.conv2(x, edge_index, edge_attr))
        if batch is not None:
            x = x.view(-1, 8 * 10)
        else:
            x = x.view(-1, 8 * 10)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 5. GAT Model
class GAT(torch.nn.Module):
    def __init__(self, heads1):
        super(GAT, self).__init__()
        self.conv1 = GATConv(1, 8, heads=heads1, dropout=0.1)
        self.conv2 = GATConv(8 * heads1, 8, heads=1, dropout=0.1)
        self.fc1 = nn.Linear(8 * 10, 16)
        self.fc2 = nn.Linear(16, 3)
        self.relu = nn.ReLU()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = data.batch if hasattr(data, 'batch') else None
        x = self.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = self.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        if batch is not None:
            x = x.view(-1, 8 * 10)
        else:
            x = x.view(-1, 8 * 10)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create graph data for GNN
pec_positions = [(0, 0), (29, 0), (0, 6), (29, 6), (3, 7), (26, 7), (9, 7), (20, 7), (15, 7)]
def create_graph_data(X):
    data_list = []
    for i in range(X.shape[0]):
        feedx = X[i, 0]
        feedx_orig = scaler_X.inverse_transform(X[i:i+1])[0, 0]
        pec_states = X[i, 1:10]
        node_features = np.concatenate([pec_states, [feedx]])
        node_features = torch.FloatTensor(node_features).view(-1, 1).to(device)
        positions = pec_positions + [(feedx_orig, 0)]
        edge_index = []
        edge_attr = []
        for j in range(10):
            for k in range(j + 1, 10):
                edge_index.append([j, k])
                edge_index.append([k, j])
                dx = positions[j][0] - positions[k][0]
                dy = positions[j][1] - positions[k][1]
                dist = np.sqrt(dx**2 + dy**2)
                edge_attr.append(dist)
                edge_attr.append(dist)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)
        edge_attr = torch.FloatTensor(edge_attr).view(-1, 1).to(device)
        data = torch_geometric.data.Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        data_list.append(data)
    return data_list

test_data_list = create_graph_data(X_test)

# Load and evaluate GCN with batched inference
gcn_model = GCN().to(device)
gcn_model.load_state_dict(torch.load('model/gcn_model.pth'))
gcn_model.eval()
test_loader = GeometricDataLoader(test_data_list, batch_size=512, shuffle=False, pin_memory=False, num_workers=0)
preds_gcn = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = gcn_model(batch)
        batch_indices = batch.batch.unique()
        preds_gcn.append(out.cpu().numpy())
preds_gcn = np.concatenate(preds_gcn, axis=0)
preds_gcn_orig = scaler_y.inverse_transform(preds_gcn)

# Load and evaluate GAT with batched inference
gat_params = best_params['gat']
gat_model = GAT(heads1=gat_params['heads1']).to(device)
gat_model.load_state_dict(torch.load('model/gat_model.pth'))
gat_model.eval()
test_loader = GeometricDataLoader(test_data_list, batch_size=512, shuffle=False, pin_memory=False, num_workers=0)
preds_gat = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = gat_model(batch)
        batch_indices = batch.batch.unique()
        preds_gat.append(out.cpu().numpy())
preds_gat = np.concatenate(preds_gat, axis=0)
preds_gat_orig = scaler_y.inverse_transform(preds_gat)

# Compute MSE for all models (in scaled space)
models = {'XGBoost': preds_xgb, 'FNN': preds_fnn, 'TabTransformer': preds_tab, 'GCN': preds_gcn, 'GAT': preds_gat}
for name, preds in models.items():
    mse = mean_squared_error(y_test, preds)
    print(f'{name} MSE: {mse:.4f}')

# Compute RMSE in original units (mm)
models_orig = {'XGBoost': preds_xgb_orig, 'FNN': preds_fnn_orig, 'TabTransformer': preds_tab_orig, 'GCN': preds_gcn_orig, 'GAT': preds_gat_orig}
for name, preds in models_orig.items():
    rmse = np.sqrt(mean_squared_error(y_test_orig, preds, multioutput='raw_values'))
    print(f'{name} RMSE (mm): Pin Distance: {rmse[0]:.4f}, Patch Length: {rmse[1]:.4f}, Pin Width: {rmse[2]:.4f}')

# Plot true vs predicted for each output (in original units)
param_names = ['Pin Distance', 'Patch Length', 'Pin Width']
y_test_orig = scaler_y.inverse_transform(y_test)
for i, param in enumerate(param_names):
    plt.figure(figsize=(10, 6))
    for name, preds in models.items():
        preds_orig = scaler_y.inverse_transform(preds)
        plt.scatter(y_test_orig[:, i], preds_orig[:, i], label=name, alpha=0.6)
    min_val = min(y_test_orig[:, i].min(), min(preds_orig[:, i].min() for preds_orig in [scaler_y.inverse_transform(preds) for preds in models.values()]))
    max_val = max(y_test_orig[:, i].max(), max(preds_orig[:, i].max() for preds_orig in [scaler_y.inverse_transform(preds) for preds in models.values()]))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='y=x')
    plt.xlabel(f'True {param} (mm)')
    plt.ylabel(f'Predicted {param} (mm)')
    plt.legend()
    plt.title(f'True vs Predicted {param}')
    plt.grid(True)
plt.show()
