import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load data
data = pd.read_csv('data/data.csv')  # region1-8, pin_pos, patch_len, pin_width
X = data.iloc[:, :8].values  # 8 binary inputs
y = data.iloc[:, 8:11].values  # 3 outputs

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
X_train, X_val, X_test = map(torch.FloatTensor, [X_train, X_val, X_test])
y_train, y_val, y_test = map(torch.FloatTensor, [y_train, y_val, y_test])

# FNN Model
class FNN(nn.Module):
    def __init__(self):
        super(FNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(),
            nn.Linear(16, 3)  # 3 outputs
        )
    def forward(self, x):
        return self.net(x)

# Train FNN
def train_fnn(X_train, y_train, X_val, y_val, epochs=200, save_path='models/fnn_model.pth'):
    model = FNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            model.eval()
            val_loss = criterion(model(X_val), y_val)
            print(f'Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    # Save the model
    torch.save(model.state_dict(), save_path)
    print(f'FNN model saved to {save_path}')
    return model

# Train and save FNN
fnn_model = train_fnn(X_train, y_train, X_val, y_val)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train.numpy(), y_train.numpy())
# Save the RF model
rf_save_path = 'models/rf_model.pkl'
joblib.dump(rf_model, rf_save_path)
print(f'RF model saved to {rf_save_path}')