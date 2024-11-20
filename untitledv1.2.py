# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 14:23:45 2023

@author: mi0025
"""




import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np

n_sensor = 19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_predictions(model, inputs, adjacency_tensor):
    # Switch model to evaluation mode
    model.eval()
    
    # Store predictions
    all_predictions = []

    with torch.no_grad(): # To save memory during inference
        for t in range(inputs.shape[0]):
            input_t = inputs[t].unsqueeze(0).to(device)
            outputs = model(input_t, adjacency_tensor)
            all_predictions.append(outputs.cpu().numpy())

    return np.vstack(all_predictions)

def processData(sensor_data, nsensor):
    #sensor_data = sensor_data.iloc[:,:n_sensor+1]
    #sensor_data_values = sensor_data
    data_mean = np.mean(sensor_data)
    data_std = np.std(sensor_data)
    sensor_data_values = (sensor_data - data_mean) / data_std
    return sensor_data_values
   

# Load your data
data = pd.read_csv("data.csv", parse_dates=["datetime"])

labels = pd.read_csv("label.csv", parse_dates=["datetime"])
data_dir = ""
distances = pd.read_csv(os.path.join(data_dir, "weights.csv"), header=None).to_numpy()

distances = torch.tensor(distances)

# Convert datetime strings to actual datetime objects
data['datetime'] = pd.to_datetime(data['datetime'])
labels['datetime'] = pd.to_datetime(labels['datetime'])

# Extract features and labels
features = torch.tensor(data = processData(data.drop(['datetime'], axis=1).values, n_sensor))  # Assuming 'datetime' is not a feature
labels = torch.tensor(labels.drop(['datetime'], axis=1).values, dtype=torch.float32)

# Calculate edge weights as the reciprocal of distances (avoiding division by zero)
edge_weights = 1.0 / torch.where(distances == 0, torch.ones_like(distances), distances)

# Create adjacency matrix with edge weights
adj_matrix = distances.clone()
adj_matrix -= torch.diag_embed(torch.diagonal(adj_matrix))

# Create a sparse adjacency matrix in COO format
edge_index = torch.nonzero(adj_matrix).t()
edge_weight = adj_matrix[edge_index[0], edge_index[1]]
adj_matrix_sparse = torch.sparse_coo_tensor(
    edge_index, edge_weight, torch.Size([features.shape[0], features.shape[0]])
)

# Convert to COO format
edge_index = adj_matrix_sparse.coalesce().indices()
edge_weight = adj_matrix_sparse.coalesce().values()

#edge_index = torch.tensor(edge_index, dtype = torch.float64)




import torch
from torch.utils.data import DataLoader, TensorDataset

#torch.cuda.set_per_process_memory_fraction(0.5)  # Adjust the fraction as needed



class ComplexGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(ComplexGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 4)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_features, edge_index, edge_weight):
        # GCN layers with attention
        x = self.conv1(node_features.to(torch.float32), edge_index, edge_weight=edge_weight.to(torch.float32))
        x, _ = self.attention(x, x, x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        x, _ = self.attention(x, x, x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        return self.sigmoid(x)





    
    
# Create the DynamicGCN model
model = ComplexGCN(input_dim=19, hidden_dim=64, output_dim=19)

# Move the entire model to GPU
model = model.to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Automatically applies sigmoid internally
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)



# Split the data into training and validation sets
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(features, labels, test_size=0.2, random_state=42)


# Move edge_index and edge_weight to the GPU
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

# Create DataLoader for training data
batch_size = 16
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)







# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Iterate over batches
    for batch_inputs, batch_labels in train_loader:
        # Forward pass
        output = model(batch_inputs.to(device), edge_index, edge_weight=edge_weight)
        loss = criterion(output, batch_labels.to(device))

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print loss every 10 epochs
    #if (epoch + 1) % 10 == 0:
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
