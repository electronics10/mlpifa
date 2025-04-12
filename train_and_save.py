# train_and_save.py
import json  # Add this import for JSON handling
import os
import platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch_geometric
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader
import xgboost as xgb
import optuna
import pickle
from tab_transformer_pytorch import TabTransformer


NUM_SAMPLES = 1500 # Input samples (80%/20%:training/testing)
PEC_POSITIONS = [(0, 0), (29, 0), (0, 6), (29, 6), (3, 7), (26, 7), (9, 7), (20, 7), (15, 7)] # Surrounding blocks postions
if NUM_SAMPLES <= 1000: BATCH_SIZE = 64
else: BATCH_SIZE = 128

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

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

# Helper function to save or update best_params.json
def save_best_params(model_name, best_params):
    best_params_file = f'model{NUM_SAMPLES}/best_params.json'
    # Ensure the model directory exists
    os.makedirs(f"model{NUM_SAMPLES}", exist_ok=True)
    
    # Load existing parameters if the file exists
    if os.path.exists(best_params_file):
        try:
            with open(best_params_file, 'r') as f:
                best_params_all = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {best_params_file}: {e}. Starting with an empty dictionary.")
            best_params_all = {}
    else:
        best_params_all = {}
    
    # Update with the new model's parameters
    best_params_all[model_name] = best_params
    
    # Save back to the file
    try:
        with open(best_params_file, 'w') as f:
            json.dump(best_params_all, f, indent=4)
        print(f"Saved {model_name} best hyperparameters to {best_params_file}")
    except IOError as e:
        print(f"Error writing to {best_params_file}: {e}")

# Load data
data = pd.read_csv('data/data.csv', header=None)
X = data.iloc[:NUM_SAMPLES, :10].values
y = data.iloc[:NUM_SAMPLES, 10:].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize inputs and outputs
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# Save scalers
os.makedirs(f"model{NUM_SAMPLES}", exist_ok=True)
with open(f'model{NUM_SAMPLES}/scaler_X.pkl', 'wb') as f:
    pickle.dump(scaler_X, f)
with open(f'model{NUM_SAMPLES}/scaler_y.pkl', 'wb') as f:
    pickle.dump(scaler_y, f)

# Convert to torch tensors and move to device
X_train_tensor = torch.FloatTensor(X_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)
y_train_tensor = torch.FloatTensor(y_train).to(device)
y_test_tensor = torch.FloatTensor(y_test).to(device)

# Prepare data for TabTransformer
categorical_columns = list(range(1, 10))
continuous_columns = [0]
X_train_cat = X_train[:, categorical_columns].astype(int)
X_test_cat = X_test[:, categorical_columns].astype(int)
X_train_cont = X_train[:, continuous_columns]
X_test_cont = X_test[:, continuous_columns]

X_train_cat_tensor = torch.LongTensor(X_train_cat).to(device)
X_test_cat_tensor = torch.LongTensor(X_test_cat).to(device)
X_train_cont_tensor = torch.FloatTensor(X_train_cont).to(device)
X_test_cont_tensor = torch.FloatTensor(X_test_cont).to(device)

# Create graph data for GNN
pec_positions = PEC_POSITIONS
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

train_data_list = create_graph_data(X_train)
test_data_list = create_graph_data(X_test)

# Dictionary to store best hyperparameters for all models
best_params_all = {}

# 1. XGBoost Hyperparameter Optimization
device_xgb = 'cpu'
print(f"XGBoost will use device: {device_xgb}")

def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'random_state': 42,
        'tree_method': 'hist',
        'device': device_xgb
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        mse_scores.append(mse)
    return np.mean(mse_scores)

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=15)
best_params_xgb = study_xgb.best_params
print("Best XGBoost params:", best_params_xgb)

# Train XGBoost with best params
best_params_xgb['tree_method'] = 'hist'
best_params_xgb['device'] = device_xgb
xgb_model = xgb.XGBRegressor(**best_params_xgb, random_state=42)
xgb_model.fit(X_train, y_train)
with open(f'model{NUM_SAMPLES}/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)

# Save XGBoost hyperparameters
save_best_params('xgboost', best_params_xgb)

# 2. FNN Hyperparameter Optimization
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

# Create dataset for FNN
fnn_dataset = TensorDataset(X_train_tensor, y_train_tensor)
fnn_test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

def objective_fnn(trial):
    hidden1 = trial.suggest_categorical('hidden1', [8, 16])
    hidden2 = trial.suggest_categorical('hidden2', [4, 8])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    lr = trial.suggest_float('lr', 0.001, 0.05, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-4, log=True)
    batch_size = BATCH_SIZE  # Adjusted for larger dataset (1700 samples)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        train_subset = TensorDataset(X_train_tensor[train_idx], y_train_tensor[train_idx])
        val_subset = TensorDataset(X_train_tensor[val_idx], y_train_tensor[val_idx])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)

        model = FNN(hidden1, hidden2, dropout_rate).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(500):
            model.train()
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        total_mse = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch)
                mse = criterion(preds, y_batch).item()
                total_mse += mse * len(X_batch)
        mse_scores.append(total_mse / len(val_subset))
    return np.mean(mse_scores)

study_fnn = optuna.create_study(direction='minimize')
study_fnn.optimize(objective_fnn, n_trials=15)
best_params_fnn = study_fnn.best_params
print("Best FNN params:", best_params_fnn)
best_params_all['fnn'] = best_params_fnn  # Store FNN params

# Train FNN with best params
fnn_model = FNN(best_params_fnn['hidden1'], best_params_fnn['hidden2'], best_params_fnn['dropout_rate']).to(device)
optimizer = torch.optim.Adam(fnn_model.parameters(), lr=best_params_fnn['lr'], weight_decay=best_params_fnn['weight_decay'])
criterion = nn.MSELoss()

train_loader = DataLoader(fnn_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=0)
for epoch in range(500):
    fnn_model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = fnn_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    if (epoch + 1) % 100 == 0:
        print(f'FNN Epoch [{epoch+1}/500], Loss: {total_loss/len(fnn_dataset):.4f}')

# Save FNN
torch.save(fnn_model.state_dict(), f'model{NUM_SAMPLES}/fnn_model.pth')

# Save FNN hyperparameters
save_best_params('fnn', best_params_fnn)

# 3. TabTransformer Hyperparameter Optimization
# Create dataset for TabTransformer
tab_dataset = TensorDataset(X_train_cat_tensor, X_train_cont_tensor, y_train_tensor)
tab_test_dataset = TensorDataset(X_test_cat_tensor, X_test_cont_tensor, y_test_tensor)

def objective_tab(trial):
    dim = trial.suggest_categorical('dim', [16, 32, 64])
    depth = trial.suggest_categorical('depth', [3, 6])
    heads = trial.suggest_categorical('heads', [4, 8])
    lr = trial.suggest_float('lr', 0.001, 0.01, log=True)
    epochs = trial.suggest_categorical('epochs', [500, 1000])
    batch_size = BATCH_SIZE  # Adjusted for larger dataset

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        train_subset = TensorDataset(X_train_cat_tensor[train_idx], X_train_cont_tensor[train_idx], y_train_tensor[train_idx])
        val_subset = TensorDataset(X_train_cat_tensor[val_idx], X_train_cont_tensor[val_idx], y_train_tensor[val_idx])
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)

        model = TabTransformer(
            categories=[2] * 9,
            num_continuous=1,
            dim=dim,
            dim_out=3,
            depth=depth,
            heads=heads,
            attn_dropout=0.1,
            ff_dropout=0.1,
            mlp_hidden_mults=(4, 2),
        ).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for epoch in range(epochs):
            model.train()
            for X_cat_batch, X_cont_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = model(X_cat_batch, X_cont_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        model.eval()
        total_mse = 0
        with torch.no_grad():
            for X_cat_batch, X_cont_batch, y_batch in val_loader:
                preds = model(X_cat_batch, X_cont_batch)
                mse = criterion(preds, y_batch).item()
                total_mse += mse * len(X_cat_batch)
        mse_scores.append(total_mse / len(val_subset))
    return np.mean(mse_scores)

study_tab = optuna.create_study(direction='minimize')
study_tab.optimize(objective_tab, n_trials=15)
best_params_tab = study_tab.best_params
print("Best TabTransformer params:", best_params_tab)

# Train TabTransformer with best params
tab_model = TabTransformer(
    categories=[2] * 9,
    num_continuous=1,
    dim=best_params_tab['dim'],
    dim_out=3,
    depth=best_params_tab['depth'],
    heads=best_params_tab['heads'],
    attn_dropout=0.1,
    ff_dropout=0.1,
    mlp_hidden_mults=(4, 2),
).to(device)
optimizer = torch.optim.Adam(tab_model.parameters(), lr=best_params_tab['lr'])
criterion = nn.MSELoss()

train_loader = DataLoader(tab_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=0)
for epoch in range(best_params_tab['epochs']):
    tab_model.train()
    total_loss = 0
    for X_cat_batch, X_cont_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = tab_model(X_cat_batch, X_cont_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_cat_batch)
    if (epoch + 1) % 100 == 0:
        print(f'TabTransformer Epoch [{epoch+1}/{best_params_tab["epochs"]}], Loss: {total_loss/len(tab_dataset):.4f}')

# Save TabTransformer
torch.save(tab_model.state_dict(), f'model{NUM_SAMPLES}/tab_model.pth')

# Save TabTransformer hyperparameters
save_best_params('tabtransformer', best_params_tab)

# 4. GCN Hyperparameter Optimization
class GCN(nn.Module):
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
            x = x.view(-1, 8 * 10)  # Reshape for batched graphs
        else:
            x = x.view(-1, 8 * 10)  # Single graph case
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def objective_gcn(trial):
    lr = trial.suggest_float('lr', 0.001, 0.01, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-4, log=True)
    batch_size = BATCH_SIZE  # Adjusted for larger dataset

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        train_data = [train_data_list[i] for i in train_idx]
        val_data = [train_data_list[i] for i in val_idx]
        y_tr = y_train_tensor[train_idx]
        y_val = y_train_tensor[val_idx]

        train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = GeometricDataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)

        model = GCN().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(500):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                batch_indices = batch.batch.unique()
                target = y_tr[batch_indices]
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

        model.eval()
        total_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                batch_indices = batch.batch.unique()
                target = y_val[batch_indices]
                mse = criterion(out, target).item()
                total_mse += mse * len(batch_indices)
        mse_scores.append(total_mse / len(val_data))
    return np.mean(mse_scores)

study_gcn = optuna.create_study(direction='minimize')
study_gcn.optimize(objective_gcn, n_trials=15)
best_params_gcn = study_gcn.best_params
print("Best GCN params:", best_params_gcn)

# Train GCN with best params
gcn_model = GCN().to(device)
optimizer = torch.optim.Adam(gcn_model.parameters(), lr=best_params_gcn['lr'], weight_decay=best_params_gcn['weight_decay'])
criterion = nn.MSELoss()

train_loader = GeometricDataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=0)
for epoch in range(500):
    gcn_model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = gcn_model(batch)
        batch_indices = batch.batch.unique()
        target = y_train_tensor[batch_indices]
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_indices)
    if (epoch + 1) % 100 == 0:
        print(f'GCN Epoch [{epoch+1}/500], Loss: {total_loss/len(train_data_list):.4f}')

# Save GCN
torch.save(gcn_model.state_dict(), f'model{NUM_SAMPLES}/gcn_model.pth')

# Save GCN hyperparameters
save_best_params('gcn', best_params_gcn)

# 5. GAT Hyperparameter Optimization
class GAT(nn.Module):
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

def objective_gat(trial):
    lr = trial.suggest_float('lr', 0.001, 0.01, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-8, 1e-4, log=True)
    heads1 = trial.suggest_categorical('heads1', [4, 8])
    batch_size = BATCH_SIZE  # Adjusted for larger dataset

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_idx, val_idx in kf.split(X_train):
        train_data = [train_data_list[i] for i in train_idx]
        val_data = [train_data_list[i] for i in val_idx]
        y_tr = y_train_tensor[train_idx]
        y_val = y_train_tensor[val_idx]

        train_loader = GeometricDataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=0)
        val_loader = GeometricDataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=False, num_workers=0)

        model = GAT(heads1).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(500):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                out = model(batch)
                batch_indices = batch.batch.unique()
                target = y_tr[batch_indices]
                loss = criterion(out, target)
                loss.backward()
                optimizer.step()

        model.eval()
        total_mse = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                batch_indices = batch.batch.unique()
                target = y_val[batch_indices]
                mse = criterion(out, target).item()
                total_mse += mse * len(batch_indices)
        mse_scores.append(total_mse / len(val_data))
    return np.mean(mse_scores)

study_gat = optuna.create_study(direction='minimize')
study_gat.optimize(objective_gat, n_trials=15)
best_params_gat = study_gat.best_params
print("Best GAT params:", best_params_gat)

# Train GAT with best params
gat_model = GAT(best_params_gat['heads1']).to(device)
optimizer = torch.optim.Adam(gat_model.parameters(), lr=best_params_gat['lr'], weight_decay=best_params_gat['weight_decay'])
criterion = nn.MSELoss()

train_loader = GeometricDataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, num_workers=0)
for epoch in range(500):
    gat_model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = gat_model(batch)
        batch_indices = batch.batch.unique()
        target = y_train_tensor[batch_indices]
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_indices)
    if (epoch + 1) % 100 == 0:
        print(f'GAT Epoch [{epoch+1}/500], Loss: {total_loss/len(train_data_list):.4f}')

# Save GAT
torch.save(gat_model.state_dict(), f'model{NUM_SAMPLES}/gat_model.pth')

# Save GAT hyperparameters
save_best_params('gat', best_params_gat)
