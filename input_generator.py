import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 150  # Adjust as needed

# Generate 8 binary input features (0 or 1)
X = np.random.randint(0, 2, (n_samples, 8))

# Create a DataFrame
data = pd.DataFrame(X, columns=[f'region{i+1}' for i in range(8)])

# Save to CSV
data.to_csv('input.csv', index=False)

print("Input data generated and saved to 'input.csv'")
