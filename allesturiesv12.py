import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from torch.nn import functional as F
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt



import matplotlib.pyplot as plt
from matplotlib import rcParams

# Set the default font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

def plot_sequence_and_error(original, reconstructed, error, threshold, title="Sequence and Reconstruction Error"):
    num_samples = min(original.shape[0], 5)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 2 * num_samples))
    feature_labels = ["Temperature", "Salinity", "DO", "Turbidity"]
    
    for i in range(num_samples):
        # Plot Original
        axes[i, 0].plot(original[i].detach().cpu().numpy(), label=feature_labels)
        axes[i, 0].set_title("Original")
        axes[i, 0].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 0].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])
        if i == 0:
            axes[i, 0].legend()

        # Plot Reconstructed
        axes[i, 1].plot(reconstructed[i].detach().cpu().numpy(), label=feature_labels)
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 1].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])

        # Plot Reconstruction Error
        axes[i, 2].plot(error[i].detach().cpu().numpy(), label="Reconstruction Error", color='red')
        axes[i, 2].axhline(threshold, color='orange', linestyle='--', label="Threshold")
        axes[i, 2].set_title("Reconstruction Error")
        axes[i, 2].set_xlim([0, len(original[i])])  # Set x-axis limits

    plt.suptitle(title)

    # Add common legend at the bottom of the plot
    lines, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)

    plt.tight_layout()

    # Save the plot with high quality
    plt.savefig('sequence_and_error_plot.png', dpi=600, bbox_inches='tight')
    
    plt.show()

# Example usage:
# Call the function with your data
# plot_sequence_and_error(original_data, reconstructed_data, error_data, threshold_value)



def plot_sequence_and_error(original, reconstructed, error, threshold, title="Sequence and Reconstruction Error"):
    num_samples = min(original.shape[0], 5)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 2 * num_samples))
    feature_labels = ["Temperature", "Salinity", "DO", "Turbidity"]
    for i in range(num_samples):
        # Plot Original
        axes[i, 0].plot(original[i].detach().cpu().numpy(), label=feature_labels)
        axes[i, 0].set_title("Original")
        axes[i, 0].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 0].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])  # Set y-axis limits
        #axes[i, 0].legend()
        if i==0:
            axes[i, 0].legend()
            

        # Plot Reconstructed
        axes[i, 1].plot(reconstructed[i].detach().cpu().numpy(), label=feature_labels)
        axes[i, 1].set_title("Reconstructed")
        axes[i, 1].set_xlim([0, len(original[i])])  # Set x-axis limits
        axes[i, 1].set_ylim([original[i].detach().cpu().numpy().min(), original[i].detach().cpu().numpy().max()])  # Set y-axis limits
        #axes[i, 1].legend()

        # Plot Reconstruction Error
        axes[i, 2].plot(error[i].detach().cpu().numpy(), label="Reconstruction Error", color='red')
        axes[i, 2].axhline(threshold, color='orange', linestyle='--', label="Threshold")
        axes[i, 2].set_title("Reconstruction Error")
        axes[i, 2].set_xlim([0, len(original[i])])  # Set x-axis limits
        #axes[i, 2].legend()

    plt.suptitle(title)
    plt.tight_layout()

    # Add common legend at the bottom of the plot
    lines, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.show()



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

    # Add common legend at the bottom of the plot
    lines, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.show()
'''
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
'''
# Function to create sequences from the data
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i: i + sequence_length].numpy()
        sequences.append(sequence)
    return np.array(sequences)

# Assuming you have initialized 'device' earlier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your sensor data and labels
data = pd.read_csv("data_df1.csv", parse_dates=["datetime"])

# Convert datetime column to datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Extract datetime column
datetime_column = data["datetime"]
# Drop datetime column for processing
data = data.drop(columns=["datetime"])

# Function to replace outliers with the mean of 5 before and 5 after values
def replace_outliers_with_nearest(df, threshold=15, window_size=5):
    df_copy = df.copy()

    for col in df_copy.columns:
        col_data = df_copy[col].values
        outliers = np.abs(col_data - col_data.mean()) > threshold * col_data.std()

        # Find indices of outliers
        outlier_indices = np.where(outliers)[0]
        threshold = 10
        window_size = 5
        

        while len(outlier_indices) > 0:
            a = len(outlier_indices)
            for index in outlier_indices:
                start_idx = max(0, index - window_size)
                end_idx = min(len(col_data), index + window_size + 1)

                # Replace outlier with the mean of 5 before and 5 after values
                df_copy.at[index, col] = np.mean(col_data[start_idx:end_idx])

            # Check for outliers again
            col_data = df_copy[col].values
            outliers = np.abs(col_data - col_data.mean()) > threshold * col_data.std()
            outlier_indices = np.where(outliers)[0]
            #print(len(outlier_indices))
            if a <= len(outlier_indices):
                window_size +=1
                

    return df_copy

# Process the data and remove outliers
data = replace_outliers_with_nearest(data)
#data.describe()

# Add datetime column back
data["datetime"] = datetime_column
#data.to_csv("noOutlierData.csv", header = True)
data = data.set_index('datetime')
# Resample the data at every two hours using mean
resampled_data = data.resample('1H').mean()

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
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden.squeeze(0)  # Squeeze to remove the batch dimension

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, output_size, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

# Define the autoencoder model using separate encoders and decoders

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_estuaries):
        super(Autoencoder, self).__init__()
        self.autoencoders = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(input_size, hidden_size, batch_first=True),
                nn.LSTM(hidden_size, input_size, batch_first=True)
            ) for _ in range(num_estuaries)
        ])

    def forward(self, x, estuary_idx):
        x, _ = self.autoencoders[estuary_idx][0](x)
        x, _ = self.autoencoders[estuary_idx][1](x)
        return x


# Assuming 'device' is defined earlier
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize and train the autoencoder for each estuary
input_size = 4  # You should set this based on your actual input size
hidden_size = 64  # You can adjust this based on your needs
num_estuaries = 19  # Number of estuaries
initial_lr = 0.01  # You can adjust this based on your needs
num_epochs = 50  # You can adjust this based on your needs
batch_size = 512

autoencoder = Autoencoder(input_size, hidden_size, num_estuaries).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=initial_lr)

# Create a DataLoader for training
train_dataset = TensorDataset(sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)

threshold =0.1

all_losses = {f'Estuary {i + 1}': [] for i in range(num_estuaries)}  # Track losses for each estuary
learning_rates = []

# Assuming 'sequences' is your input data
for epoch in range(num_epochs):
    epoch_losses = {f'Estuary {i + 1}': [] for i in range(num_estuaries)}  # Track losses for each estuary
    for estuary_idx in range(num_estuaries):
        for batch_inputs in train_loader:
            batch_inputs = batch_inputs[0].to(device)

            optimizer.zero_grad()
            # Extract the current estuary sequence
            current_estuary_sequence = batch_inputs[:, :, estuary_idx * 4: (estuary_idx + 1) * 4]
            outputs = autoencoder(current_estuary_sequence, estuary_idx)
            
            loss = criterion(outputs, current_estuary_sequence)
            loss.backward()
            optimizer.step()

            epoch_losses[f'Estuary {estuary_idx + 1}'].append(loss.item())

        # Print the loss for each epoch and estuary
        print(f'Epoch [{epoch + 1}/{num_epochs}], Estuary {estuary_idx + 1}, Loss: {loss.item():.4f}')
        
        if epoch % 5  == 0:
            with torch.no_grad():
            #for estuary_idx in range(num_estuaries):
                reconstructed_samples = autoencoder(current_estuary_sequence, estuary_idx)
                errors = F.mse_loss(reconstructed_samples, current_estuary_sequence, reduction='none').mean(dim=2)
                mean_errors = errors.mean(dim=1)
    
                normal_indices = torch.where(mean_errors <= mean_errors.mean())[0]
                abnormal_indices = torch.where(mean_errors > mean_errors.mean()*2)[0]
    
                if len(normal_indices) > 1:
                    normal_samples = current_estuary_sequence[normal_indices]
                    normal_reconstructed = reconstructed_samples[normal_indices]
                    normal_errors = errors[normal_indices]
                    plot_sequence_and_error(normal_samples, normal_reconstructed, normal_errors, threshold,
                                            title=f"Epoch {epoch} Estuary: WeeksBay Normal Sequences")
                #{estuary_idx + 1}
                if len(abnormal_indices) > 1:
                    abnormal_samples = current_estuary_sequence[abnormal_indices]
                    abnormal_reconstructed = reconstructed_samples[abnormal_indices]
                    abnormal_errors = errors[abnormal_indices]
                    plot_sequence_and_error(abnormal_samples, abnormal_reconstructed, abnormal_errors, threshold,
                                            title=f"Epoch {epoch} Estuary: WeeksBay  Abnormal Sequences")
# Calculate the average loss for the epoch for each estuary
    for estuary_idx, estuary_loss in epoch_losses.items():
        average_estuary_loss = sum(estuary_loss) / len(estuary_loss)
        all_losses[estuary_idx].append(average_estuary_loss)

    # Dynamic learning rate adjustment
    if epoch % 5 == 0 and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.9
    current_lr = optimizer.param_groups[0]['lr']
    learning_rates.append(current_lr)
           
            
            
            
# Plot the training losses for each estuary
for estuary_idx, estuary_losses in all_losses.items():
    plt.plot(estuary_losses, label=estuary_idx)

    plt.title("Training Loss Over Epochs for Each Estuary")
    plt.xlabel("Epochs")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.show()

# Plotting the learning rates
plt.plot(learning_rates, marker='o')
plt.title("Learning Rate Adjustment Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.show()

###
import plotly.graph_objects as go
import plotly.express as px

# Assuming 'all_losses' and 'learning_rates' are defined in your environment

# Plot the training losses for each estuary
fig_losses = go.Figure()

for estuary_idx, estuary_losses in all_losses.items():
    fig_losses.add_trace(go.Scatter(
        x=list(range(1, len(estuary_losses) + 1)),
        y=estuary_losses,
        mode='lines',
        name=str(estuary_idx),
        line=dict(width=2, backoff=0.7)
    ))

fig_losses.update_layout(
    title="Training Loss Over Epochs for Each Estuary",
    xaxis_title="Epochs",
    yaxis_title="Average Loss",
    legend=dict(orientation="h", x=0.5, y=-0.15),
    font=dict(family="Times New Roman"),
)

# Save the interactive plot
fig_losses.write_html("training_losses_plot.html", full_html=False)

# Plotting the learning rates
fig_learning_rates = px.line(
    x=list(range(1, len(learning_rates) + 1)),
    y=learning_rates,
    markers=True,
    labels={"x": "Epochs", "y": "Learning Rate"},
    title="Learning Rate Adjustment Over Epochs",
)

fig_learning_rates.update_layout(
    font=dict(family="Times New Roman"),
)

# Save the interactive plot
fig_learning_rates.write_html("learning_rates_plot.html", full_html=False)


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import numpy as np

# Assuming 'all_losses' and 'learning_rates' are defined in your environment

# Create a subplot for the training losses
fig_losses = make_subplots(rows=len(all_losses), cols=1, subplot_titles=[f"Estuary {idx}" for idx in all_losses.keys()])

for i, (estuary_idx, estuary_losses) in enumerate(all_losses.items()):
    fig_losses.add_trace(go.Scatter(
        x=np.arange(1, len(estuary_losses) + 1),
        y=estuary_losses,
        mode='lines',
        name=f'Estuary {estuary_idx}',
        line=dict(width=2, backoff=0.7)
    ), row=i+1, col=1)

fig_losses.update_layout(
    title_text="Training Loss Over Epochs for Each Estuary",
    xaxis_title="Epochs",
    yaxis_title="Average Loss",
    showlegend=False,  # Legends are added manually below
    font=dict(family="Times New Roman"),
)

# Add legends manually at the bottom
fig_losses.add_trace(go.Scatter(), row=len(all_losses), col=1, name=', '.join(all_losses.keys()))

# Create a subplot for the learning rates
fig_learning_rates = go.Figure()

fig_learning_rates.add_trace(go.Scatter(
    x=np.arange(1, len(learning_rates) + 1),
    y=learning_rates,
    mode='markers+lines',
    marker=dict(symbol='circle', size=8),
    line=dict(width=2, backoff=0.7)
))

fig_learning_rates.update_layout(
    title="Learning Rate Adjustment Over Epochs",
    xaxis_title="Epochs",
    yaxis_title="Learning Rate",
    font=dict(family="Times New Roman"),
)

# Save the interactive plots as JPEG
fig_losses.write_image("training_losses_plot.jpg", scale=6)  # 600 DPI
fig_learning_rates.write_image("learning_rates_plot.jpg", scale=6)  # 600 DPI



import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Assuming 'all_losses' and 'learning_rates' are defined in your environment

# Create a figure for the training losses
fig_losses, ax_losses = plt.subplots(figsize=(10, 6))

#for estuary_idx, estuary_losses in all_losses.items():
ax_losses.plot(estuary_losses, label=f"Estuary {estuary_idx}", alpha=0.7)

ax_losses.set_title("Training Loss Over Epochs for Estuary: Weeksbay", fontsize=14, fontname="Times New Roman")
ax_losses.set_xlabel("Epochs", fontsize=12, fontname="Times New Roman")
ax_losses.set_ylabel("Average Loss", fontsize=12, fontname="Times New Roman")
ax_losses.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=len(all_losses), fontsize=10, frameon=False)
ax_losses.xaxis.set_major_locator(MaxNLocator(integer=True))
ax_losses.yaxis.grid(True, linestyle='--', alpha=0.7)

# Save the training losses plot as JPEG
fig_losses.savefig("training_losses_plot.jpg", dpi=600, bbox_inches='tight')

# Create a figure for the learning rates
fig_lr, ax_lr = plt.subplots(figsize=(10, 6))

ax_lr.plot(learning_rates, marker='o', alpha=0.7)

ax_lr.set_title("Learning Rate Adjustment Over Epochs", fontsize=14, fontname="Times New Roman")
ax_lr.set_xlabel("Epochs", fontsize=12, fontname="Times New Roman")
ax_lr.set_ylabel("Learning Rate", fontsize=12, fontname="Times New Roman")
ax_lr.yaxis.grid(True, linestyle='--', alpha=0.7)

# Save the learning rates plot as JPEG
fig_lr.savefig("learning_rates_plot.jpg", dpi=600, bbox_inches='tight')

plt.show()




# Assuming 'autoencoder' is your trained model
torch.save(autoencoder.state_dict(), 'autoencoder_modelV19Estuaries.pth')
# Assuming 'Autoencoder' is the class you defined for your model
model = Autoencoder(input_size, hidden_size, num_estuaries)
model.load_state_dict(torch.load('autoencoder_modelV19Estuaries.pth'))
model.to(device)


# Function to find disturbance information
def find_disturbance_info(normal_range, abnormal_sequence):
    # Assume normal_range and abnormal_sequence are NumPy arrays
    start_time = None
    peak_time = None
    restoration_time = None
    end_time = None

    # Find the start time
    start_time_idx = np.argmax(abnormal_sequence[:, 0] > normal_range[0])
    start_time = abnormal_sequence[start_time_idx, -1]

    # Keep track of peak deviation
    peak_deviation = 0

    # Find peak time
    for i in range(len(abnormal_sequence)):
        deviation = np.abs(abnormal_sequence[i, :] - normal_range)
        max_deviation = np.max(deviation)

        if max_deviation > peak_deviation:
            peak_deviation = max_deviation
            peak_time = abnormal_sequence[i, -1]

    # Find restoration time
    restoration_time_idx = np.argmax(abnormal_sequence[:, 0] < normal_range[0])
    restoration_time = abnormal_sequence[restoration_time_idx, -1]

    # Find end time
    end_time = abnormal_sequence[-1, -1]

    disturbance_duration = peak_time - start_time if start_time is not None and peak_time is not None else None
    restoration_duration = restoration_time - peak_time if peak_time is not None and restoration_time is not None else None

    return start_time, peak_time, disturbance_duration, restoration_duration, end_time



# Initialize the dataframe

df = pd.DataFrame()
df2 = pd.DataFrame()

# Use the trained model to obtain reconstruction errors for all estuaries
for estuary_idx in range(num_estuaries):
    # Extract data for the current estuary
    estuary_data = sequences[:, :, estuary_idx * 4: (estuary_idx + 1) * 4]

    # Assuming you have the time indices
    time_indices = resampled_data.index

    # Use the model to obtain reconstruction errors
    with torch.no_grad():
        reconstructed_samples = model(estuary_data, estuary_idx)
        errors = F.mse_loss(reconstructed_samples, estuary_data, reduction='none').mean(dim=2)
        mean_errors = errors.mean(dim=1)

    # Find disturbance start indices
    #threshold = mean_errors.std() * 5
    
    reconstruction_errors = mean_errors.detach().cpu().numpy()

    # If using NumPy arrays
    #reconstruction_errors = np.array(reconstruction_errors)
        
    normalized_errors = (reconstruction_errors - np.min(reconstruction_errors)) / (np.max(reconstruction_errors) - np.min(reconstruction_errors))
    
    anomalies = [1 if error > 1*normalized_errors.std() else 0 for error in normalized_errors]
    
    df[f'Estuary_{estuary_idx + 1}'] = normalized_errors
    df2[f'Estuary_{estuary_idx + 1}'] = anomalies
    
df.index = resampled_data.index[sequence_length-1:]
df2.index = df.index
# Save the dataframe to a CSV file
df.to_csv('reconstruction_errorsv2.csv', index=True)
df2.to_csv('abnormallabelsvstd2.csv', index=True)

'''

def merge_series(arr, threshold):
    # Find unique numbers in the array (excluding 0)
    unique_numbers = list(set(filter(lambda x: x != 0, arr)))

    # Initialize a dictionary to store the start and end indices for each unique number
    indices_dict = {num: {'start': None, 'end': None} for num in unique_numbers}

    # Iterate through the array to find the start and end indices for each unique number
    for i, num in enumerate(arr):
        if num != 0:
            if indices_dict[num]['start'] is None:
                indices_dict[num]['start'] = i
            indices_dict[num]['end'] = i

    # Merge series based on the specified threshold
    for i in range(len(unique_numbers) - 1):
        current_num = unique_numbers[i]
        next_num = unique_numbers[i + 1]

        end_index_current = indices_dict[current_num]['end']
        start_index_next = indices_dict[next_num]['start']
        end_index_next = indices_dict[next_num]['end']

        if start_index_next - end_index_current <= threshold:
            # Merge the series by updating the elements in the specified range
            arr[end_index_current + 1:end_index_next+1] = [current_num] * (end_index_next - end_index_current)

    return arr

# Assuming df is your original DataFrame

def normalize_and_add_count(df, label_column, min_series_length=6, max_gap_between_series=4):
    df[label_column] = 0

    # Identify the start of a new series
    start_of_series = (df[label_column] == 1) & (df[label_column].shift(1) == 0)

    # Cumulatively count series
    df['hurricaneCount'] = start_of_series.cumsum()
    df.loc[df[label_column] == 0, 'hurricaneCount'] = 0

    # Calculate the length of each series
    series_lengths = df.groupby('hurricaneCount').size()

    # Identify and drop series with fewer than min_series_length observations
    short_series = series_lengths[series_lengths < min_series_length].index
    df = df[~df['hurricaneCount'].isin(short_series)]
    input_array = np.array(df['hurricaneCount'])

    result_array = merge_series(input_array, max_gap_between_series)
    df['newLabel'] = result_array

    return df[['newLabel']]  # Return only the newLabel column

data = pd.read_csv("abnormallabels.csv", parse_dates=["datetime"])

# Convert datetime column to datetime format
data['datetime'] = pd.to_datetime(data['datetime'])

# Extract datetime column
datetime_column = data["datetime"]
# Drop datetime column for processing
data = data.drop(columns=["datetime"])
# List of labels from label1 to label19
labels = data.columns

# Create a new DataFrame for the results
result_df = pd.DataFrame()

# Apply the function for each label
for label in labels:
    result_df[label] = normalize_and_add_count(data.copy(), label)['newLabel']
    
result_df.index = datetime_column

# result_df now contains newLabel1 to newLabel19


'''

'''



# Initialize the dataframe
columns = []
for estuary_idx in range(num_estuaries):
    for sensor_idx in range(4):
        columns.append(f'Estuary_{estuary_idx + 1}_Sensor_{sensor_idx + 1}')

columns += ['Datetime', 'Reconstruction_Error']

df = pd.DataFrame(columns=columns)

# Use the trained model to obtain reconstruction errors for all estuaries
for estuary_idx in range(num_estuaries):
    # Extract data for the current estuary
    estuary_data = sequences[:, :, estuary_idx * 4: (estuary_idx + 1) * 4]

    # Assuming you have the time indices
    time_indices = resampled_data.index

    # Use the model to obtain reconstruction errors
    with torch.no_grad():
        reconstruction_errors = []
        reconstructed_samples = model(estuary_data, estuary_idx)
        errors = F.mse_loss(reconstructed_samples, estuary_data, reduction='none').mean(dim=2)
        mean_errors = errors.mean(dim=1)

    # Find disturbance start indices
    disturbance_start_indices = torch.where(mean_errors > mean_errors.std()*5)[0].detach().cpu().numpy()

    # Initialize disturbance durations and peaks
    disturbance_durations = []
    disturbance_peaks = []

    # Iterate over disturbance start indices
    for start_index in disturbance_start_indices:
        # Extract reconstruction errors after the disturbance start
        disturbance_errors = mean_errors[start_index:]

        # Define a criterion for ending disturbance phase
        end_index = np.argmax((disturbance_errors < mean_errors.std()*5).detach().cpu().numpy())

        if end_index == 0:
            # If the error doesn't decrease significantly, consider the disturbance to last until the end of the sequence
            end_index = len(disturbance_errors)

        disturbance_durations.append(end_index)

        # Find the index of the peak error
        peak_index = np.argmax(disturbance_errors.detach().cpu().numpy())

        # Add the index relative to the start of the disturbance
        disturbance_peaks.append(start_index + peak_index)

    # Plot the original data with disturbance phases
    original_data = estuary_data.cpu().numpy().reshape(-1, 4)
    reconstruction_error = mean_errors.detach().cpu().numpy()

    # Add data to the dataframe
    data_for_df = original_data.flatten().tolist()
    data_for_df += time_indices
    data_for_df += reconstruction_error.tolist()

    df = pd.concat([df, pd.DataFrame([data_for_df], columns=columns)], ignore_index=True)

# Save the dataframe to a CSV file
df.to_csv('reconstruction_errors.csv', index=False)






# Assuming you have a reconstruction_error array containing errors for each sequence
disturbance_threshold = 0.1  # Adjust this threshold based on your data

disturbance_start_indices = np.where(reconstruction_error > disturbance_threshold)[0]





# Anomaly detection
autoencoder.eval()
with torch.no_grad():
    reconstruction_errors = []
    for i in range(len(sequences)):
        data_point = sequences[i].to(device)
        output = autoencoder(data_point.unsqueeze(0))
        loss = criterion(output, data_point.unsqueeze(0))
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
anomalies = [1 if error > 0.2 else 0 for error in normalized_errors]
print(anomalies)


#labels = pd.read_csv("label_df1.csv", header=None)
labels = pd.read_csv("label_df1.csv", parse_dates=["datetime"])
# Convert datetime strings to actual datetime objects
labels = labels.iloc[:, :2]

labels['datetime'] = pd.to_datetime(labels['datetime'])

labels = labels.set_index('datetime')


# Resample the data at every two hours using mean
resampled_labels = labels.resample('1H').mean()

# Extracting the second column as a NumPy array
label1 = resampled_labels.iloc[:, 0].to_numpy()
label1 = [1 if error > 0.1 else 0 for error in label1]


# Now you have trained autoencoders for all 19 estuaries




# Initialize and train the autoencoder
input_size = 4  # Number of features
hidden_size = 8  # You can adjust this based on your needs
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

# Anomaly detection loop
for epoch in range(num_epochs):
    epoch_losses = {f'Estuary {i + 1}': [] for i in range(num_estuaries)}  # Track losses for each estuary
    for estuary_idx in range(num_estuaries):
        for batch_inputs in train_loader:
            batch_inputs = batch_inputs[0].to(device)

            optimizer.zero_grad()
            # Extract the current estuary sequence
            current_estuary_sequence = batch_inputs[:, :, estuary_idx * 4: (estuary_idx + 1) * 4]
            outputs = autoencoder(current_estuary_sequence, estuary_idx)
            
            loss = criterion(outputs, current_estuary_sequence)
            loss.backward()
            optimizer.step()

            epoch_losses[f'Estuary {estuary_idx + 1}'].append(loss.item())

        # Print the loss for each epoch and estuary
        print(f'Epoch [{epoch + 1}/{num_epochs}], Estuary {estuary_idx + 1}, Loss: {loss.item():.4f}')

    

    #print(f'Epoch [{epoch + 1}/{num_epochs}], Average Losses: {epoch_losses}')

    # Plot normal sequences every 2 epochs
    # Plot normal sequences every 2 epochs
        if epoch % 5  == 0:
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

'''