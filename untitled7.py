import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Assuming you have initialized 'device' earlier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your sensor data and labels
data = pd.read_csv("data_df1.csv", parse_dates=["datetime"])
# Convert datetime strings to actual datetime objects
data = data.iloc[:, :5]

data['datetime'] = pd.to_datetime(data['datetime'])

data = data.set_index('datetime')


# Resample the data at every two hours using mean
resampled_data = data.resample('2H').mean()

# Normalize the resampled data
scaler = StandardScaler()
resampled_data_normalized = scaler.fit_transform(resampled_data)

# Set parameters
sequence_length = 10
hidden_size = 8
threshold = 0.1
num_epochs = 50
batch_size = 32
initial_lr = 0.001

# Function to create sequences from the data
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i : i + sequence_length].numpy()
        sequences.append(sequence)
    return np.array(sequences)

# Convert data to PyTorch tensor
data_tensor = torch.tensor(resampled_data_normalized, dtype=torch.float32)

# Create sequences
sequences = create_sequences(data_tensor, sequence_length)

# Convert sequences to PyTorch tensor
sequences = torch.tensor(sequences, dtype=torch.float32)

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        x, _ = self.encoder(x)
        x, _ = self.decoder(x)
        return x

sequences = sequences.to(device)

# Initialize and train the autoencoder
input_size = sequences.size(-1)  # Number of features
autoencoder = Autoencoder(input_size, hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=initial_lr)

# Create a DataLoader for training
train_dataset = TensorDataset(sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop with dynamic learning rate adjustment
for epoch in range(num_epochs):
    for batch_inputs in train_loader:
        batch_inputs = batch_inputs[0].to(device)

        optimizer.zero_grad()
        outputs = autoencoder(batch_inputs)
        loss = criterion(outputs, batch_inputs)
        loss.backward()
        optimizer.step()

    # Dynamic learning rate adjustment
    if epoch % 10 == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Anomaly detection
autoencoder.eval()
with torch.no_grad():
    reconstruction_errors = []
    for batch_inputs in train_loader:
        batch_inputs = batch_inputs[0].to(device)
        outputs = autoencoder(batch_inputs)
        loss = criterion(outputs, batch_inputs)
        reconstruction_errors.append(loss.item())
        
# Anomaly detection
autoencoder.eval()
with torch.no_grad():
    reconstruction_errors = []
    for i in range(len(sequences)):  # Assuming 'dataset' is your PyTorch dataset
        data_point = sequences[i].to(device)  # Assuming your data is the first element in each tuple
        output = autoencoder(data_point.unsqueeze(0))  # Unsqueeze to add batch dimension
        loss = criterion(output, data_point.unsqueeze(0))
        reconstruction_errors.append(loss.item())

# If using PyTorch tensors directly
# reconstruction_errors = torch.tensor(reconstruction_errors).cpu().numpy()

# If using NumPy arrays
reconstruction_errors = np.array(reconstruction_errors)
    
normalized_errors = (reconstruction_errors - np.min(reconstruction_errors)) / (np.max(reconstruction_errors) - np.min(reconstruction_errors))
       
     
        
# Set a threshold for anomaly detection
anomalies = [1 if error > 0.05 else 0 for error in normalized_errors]
print(anomalies)


#labels = pd.read_csv("label_df1.csv", header=None)
labels = pd.read_csv("label_df1.csv", parse_dates=["datetime"])
# Convert datetime strings to actual datetime objects
labels = labels.iloc[:, :2]

labels['datetime'] = pd.to_datetime(labels['datetime'])

labels = labels.set_index('datetime')


# Resample the data at every two hours using mean
resampled_labels = labels.resample('2H').mean()

# Extracting the second column as a NumPy array
label1 = resampled_labels.iloc[:, 0].to_numpy()
label1 = [1 if error > 0.5 else 0 for error in label1]
    
  

from sklearn.metrics import confusion_matrix

confusion_matrix(label1[9:], anomalies)


print('Confusion Matrix:')
print(conf_matrix)