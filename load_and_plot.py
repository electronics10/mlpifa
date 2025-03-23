import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Load model path
fnn_path = 'models/fnn_model.pth'
rf_path = 'models/rf_model.pkl'

# Load data (same as training for consistency)
data = pd.read_csv('data/data.csv')  # region1-8, pin_pos, patch_len, pin_width
X = data.iloc[:, :8].values  # 8 binary inputs
y = data.iloc[:, 8:11].values  # 3 outputs

# Split (same random_state to match training split)
_, X_temp, _, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
X_val, X_test = map(torch.FloatTensor, [X_val, X_test])
y_val, y_test = map(torch.FloatTensor, [y_val, y_test])

# FNN Model (must match training architecture)
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 3)  # 3 outputs
        )
    def forward(self, x):
        return self.net(x)

# Load FNN
def load_fnn_model(save_path=fnn_path):
    model = FNN()
    model.load_state_dict(torch.load(save_path))
    model.eval()
    print(f'FNN model loaded from {save_path}')
    return model

# Load RF
def load_rf_model(save_path=rf_path):
    model = joblib.load(save_path)
    print(f'RF model loaded from {save_path}')
    return model

# Evaluate
def evaluate(model, X_test, y_test, model_type='fnn'):
    if model_type == 'fnn':
        model.eval()
        with torch.no_grad():
            preds = model(X_test)
    else:
        preds = torch.FloatTensor(model.predict(X_test.numpy()))
    mse = mean_squared_error(y_test.numpy(), preds.numpy())
    return mse, preds

# Load models
fnn_loaded = load_fnn_model(fnn_path)
rf_loaded = load_rf_model(rf_path)

# Evaluate loaded models
mse_fnn, preds_fnn = evaluate(fnn_loaded, X_test, y_test, 'fnn')
mse_rf, preds_rf = evaluate(rf_loaded, X_test, y_test, 'rf')
print(f'Loaded FNN MSE: {mse_fnn:.4f}')
print(f'Loaded RF MSE: {mse_rf:.4f}')

# Plot all 3 parameters
for i, param in enumerate(['Pin Position', 'Patch Length', 'Pin Width']):
    plt.figure()
    plt.scatter(y_test[:, i], preds_fnn[:, i], label='FNN', alpha=0.6)
    plt.scatter(y_test[:, i], preds_rf[:, i], label='RF', alpha=0.6)
    plt.xlabel(f'True {param}')
    plt.ylabel(f'Predicted {param}')
    plt.legend()
    plt.title(f'True vs Predicted {param}')
    # Add y=x reference line
    pmin = min(min(preds_fnn[:, i]), min(preds_rf[:, i]))
    pmax = max(max(preds_fnn[:, i]), max(preds_rf[:, i]))
    plt.plot([pmin, pmax], [pmin, pmax], 'g--', label='y=x')  
plt.show()