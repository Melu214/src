import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from torch.nn import functional as F

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import RobustScaler




# Function to plot sequences and reconstruction error
def plot_sequence_and_error(original, reconstructed, error, threshold, title="Sequence and Reconstruction Error"):
    num_samples = min(original.shape[0], 5)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 2 * num_samples))

    for i in range(num_samples):
        # Plot Original
        axes[i, 0].plot(original[i].detach().cpu().numpy(), label="Original")
        axes[i, 0].set_title("Original")
        axes[i, 0].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 0].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])  # Set y-axis limits

        # Plot Reconstructed
        axes[i, 1].plot(reconstructed[i].detach().cpu().numpy(), label="Reconstructed")
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 1].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])  # Set y-axis limits

        # Plot Reconstruction Error
        axes[i, 2].plot(error[i].detach().cpu().numpy(), label="Reconstruction Error", color='red')
        axes[i, 2].axhline(threshold, color='orange', linestyle='--', label="Threshold")
        axes[i, 2].set_title("Reconstruction Error")
        axes[i, 2].set_xlim([0, len(original[i])])  # Set x-axis limits

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Function to create sequences from the data
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i: i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

# Assuming you have initialized 'device' earlier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your sensor data and labels
data = pd.read_csv("data_df1.csv", parse_dates=["datetime"])

# Convert datetime column to datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Extract datetime column
#datetime_column = data["datetime"]
# Drop datetime column for processing
#data = data.drop(columns=["datetime"])

# Function to replace outliers with the mean of 5 before and 5 after values
def replace_outliers_with_nearest(df, threshold=10, window_size=5):
    df_copy = df.copy()

    for col in df_copy.columns:
        col_data = df_copy[col].values
        outliers = np.abs(col_data - col_data.mean()) > threshold * col_data.std()

        # Find indices of outliers
        outlier_indices = np.where(outliers)[0]

        while len(outlier_indices) > 0:
            for index in outlier_indices:
                start_idx = max(0, index - window_size)
                end_idx = min(len(col_data), index + window_size + 1)

                # Replace outlier with the mean of 5 before and 5 after values
                df_copy.at[index, col] = np.mean(col_data[start_idx:end_idx])

            # Check for outliers again
            col_data = df_copy[col].values
            outliers = np.abs(col_data - col_data.mean()) > threshold * col_data.std()
            outlier_indices = np.where(outliers)[0]

    return df_copy

# Process the data and remove outliers
#data = replace_outliers_with_nearest(data)

# Add datetime column back
#data["datetime"] = datetime_column
data = data.set_index('datetime')
# Resample the data at every two hours using mean
resampled_data = data.resample('5H').mean()

# Normalize the resampled data
scaler = RobustScaler()  # RobustScaler is less sensitive to outliers
resampled_data_normalized = scaler.fit_transform(resampled_data)

# Convert data to PyTorch tensor
data_tensor = torch.tensor(resampled_data_normalized, dtype=torch.float32)

# Create sequences
sequence_length = 10  # You can adjust this based on your needs
sequences = create_sequences(data_tensor, sequence_length)

# Convert sequences to PyTorch tensor
sequences = torch.tensor(sequences, dtype=torch.float32)

# Assuming 'device' is defined earlier
sequences = sequences.to(device)

# Define a single shared encoder for all estuaries
class SharedEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SharedEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden.squeeze(0)  # Squeeze to remove the batch dimension

# Define the autoencoder model using the shared encoder
class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_estuaries):
        super(Autoencoder, self).__init__()
        self.shared_encoder = SharedEncoder(input_size, hidden_size)
        self.decoders = nn.ModuleList([nn.LSTM(hidden_size, input_size, batch_first=True) for _ in range(num_estuaries)])

    def forward(self, x, estuary_idx):
        encoded = self.shared_encoder(x)
        decoded, _ = self.decoders[estuary_idx](encoded.unsqueeze(1).repeat(1, x.size(1), 1))
        return decoded

# Initialize and train the autoencoder
input_size = sequences.size(-1)  # Number of features
hidden_size = 64  # You can adjust this based on your needs
initial_lr = 0.001  # You can adjust this based on your needs
num_estuaries = 19  # Number of estuaries
batch_size =32
autoencoder = Autoencoder(input_size, hidden_size, num_estuaries).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=initial_lr)

# Create a DataLoader for training
train_dataset = TensorDataset(sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 500  # You can adjust this based on your needs
threshold = .5  # You can adjust this based on your needs
import matplotlib.pyplot as plt
all_losses = {f'Estuary {i + 1}': [] for i in range(num_estuaries)}  # Track losses for each estuary

# Anomaly detection loop

for epoch in range(num_epochs):
    epoch_losses = {f'Estuary {i + 1}': [] for i in range(num_estuaries)}  # Track losses for each estuary
    for estuary_idx in range(num_estuaries):
        for batch_inputs in train_loader:
            batch_inputs = batch_inputs[0].to(device)

            optimizer.zero_grad()
            outputs = autoencoder(batch_inputs, estuary_idx)
            loss = criterion(outputs, batch_inputs)
            loss.backward()
            optimizer.step()

            epoch_losses[f'Estuary {estuary_idx + 1}'].append(loss.item())

        # Print the loss for each epoch and estuary
        print(f'Epoch [{epoch + 1}/{num_epochs}], Estuary {estuary_idx + 1}, Loss: {loss.item():.4f}')

    

    #print(f'Epoch [{epoch + 1}/{num_epochs}], Average Losses: {epoch_losses}')

    # Plot normal sequences every 2 epochs
    # Plot normal sequences every 2 epochs
        if epoch % 20  == 0:
            with torch.no_grad():
            #for estuary_idx in range(num_estuaries):
                reconstructed_samples = autoencoder(sequences, estuary_idx)
                errors = F.mse_loss(reconstructed_samples, sequences, reduction='none').mean(dim=2)
                mean_errors = errors.mean(dim=1)
    
                normal_indices = torch.where(mean_errors <= threshold/10)[0]
                abnormal_indices = torch.where(mean_errors > threshold*50)[0]
    
                if len(normal_indices) > 0:
                    normal_samples = sequences[normal_indices, :, estuary_idx * 4: (estuary_idx + 1) * 4]
                    normal_reconstructed = reconstructed_samples[normal_indices, :, estuary_idx * 4: (estuary_idx + 1) * 4]
                    normal_errors = errors[normal_indices]
                    plot_sequence_and_error(normal_samples, normal_reconstructed, normal_errors, threshold,
                                            title=f"Epoch {epoch} Estuary {estuary_idx + 1} Normal Sequences")
    
                if len(abnormal_indices) > 0:
                    abnormal_samples = sequences[abnormal_indices, :, estuary_idx * 4: (estuary_idx + 1) * 4]
                    abnormal_reconstructed = reconstructed_samples[abnormal_indices, :, estuary_idx * 4: (estuary_idx + 1) * 4]
                    abnormal_errors = errors[abnormal_indices]
                    plot_sequence_and_error(abnormal_samples, abnormal_reconstructed, abnormal_errors, threshold,
                                            title=f"Epoch {epoch} Estuary {estuary_idx + 1} Abnormal Sequences")
# Calculate the average loss for the epoch for each estuary
    for estuary_idx, estuary_loss in epoch_losses.items():
        average_estuary_loss = sum(estuary_loss) / len(estuary_loss)
        all_losses[estuary_idx].append(average_estuary_loss)

    # Dynamic learning rate adjustment
    if epoch % 5 == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
            
            
            
            
            
# Plot the training losses for each estuary
for estuary_idx, estuary_losses in all_losses.items():
    plt.plot(estuary_losses, label=estuary_idx)

plt.title("Training Loss Over Epochs for Each Estuary")
plt.xlabel("Epochs")
plt.ylabel("Average Loss")
plt.legend()
plt.show()