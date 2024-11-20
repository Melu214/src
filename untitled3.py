import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split


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
features = torch.tensor(data = processData(data.drop(['datetime'], axis=1).values, n_sensor), dtype=torch.float32)  # Assuming 'datetime' is not a feature
labels = torch.tensor(labels.drop(['datetime'], axis=1).values, dtype=torch.float32)

# Calculate edge weights as the reciprocal of distances (avoiding division by zero)
distances = distances.to(torch.float32)
edge_weight = 1.0 / torch.where(distances == 0, torch.ones_like(distances), distances)

# Create adjacency matrix with edge weights
adj_matrix = distances.clone()
adj_matrix -= torch.diag_embed(torch.diagonal(adj_matrix))

# Create a sparse adjacency matrix in COO format
edge_index = torch.nonzero(adj_matrix, as_tuple=False).t()
edge_weight = edge_weight[edge_index[0], edge_index[1]].to(torch.float32)
adj_matrix_sparse = torch.sparse_coo_tensor(
    edge_index, edge_weight, torch.Size([features.shape[0], features.shape[0]])
    )

X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(features, labels, test_size=0.2, random_state=42)



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
        x = self.conv1(node_features, edge_index, edge_weight=edge_weight.to(torch.float32))
        x, _ = self.attention(x, x, x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        x, _ = self.attention(x, x, x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        return self.sigmoid(x)




hidden_features = 64
base_learning_rate = 0.01
# Model, Loss and Optimizer
# Create the ComplexGCN model
model = ComplexGCN(input_dim=19, hidden_dim=hidden_features, output_dim=19)
model = model.to(device)
# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=base_learning_rate)




# Early stopping parameters
patience = 10
best_loss = float('inf')
counter = 0

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, verbose=True)
# Assuming 'inputs' and 'labels' are your data tensors
inputs = torch.tensor(features, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

# Split data for training and validation
num_samples = len(inputs)
train_size = int(0.9 * num_samples)
val_size = num_samples - train_size

# Create Tensor datasets
train_dataset = TensorDataset(inputs[:train_size], labels[:train_size])
val_dataset = TensorDataset(inputs[train_size:], labels[train_size:])

# Create DataLoader
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


epochs = 50
lossHist = []
valLossHist = []
testLossHist = []
for epoch in range(epochs):
    model.train()  # Switch to training mode
    total_loss = 0
    
    for batch_inputs, batch_labels in train_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs, edge_index, edge_weight)
        loss = criterion(outputs, batch_labels)
        #l2_loss = model.l2_regularization_loss()
        total_loss = loss # + l2_loss
        total_loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_inputs.size(0)
    
    avg_loss = total_loss / len(train_dataset)
    lossHist.append(avg_loss)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            edge_index = edge_index.to(device)
            edge_weight = edge_weight.to(device)
            outputs = model(batch_inputs, edge_index, edge_weight)
            loss = criterion(outputs, batch_labels)
            val_loss += loss.item() * batch_inputs.size(0)

    avg_val_loss = val_loss / len(val_dataset)
    valLossHist.append(avg_val_loss)
    
    # Learning rate adjustment
    scheduler.step(avg_val_loss)

    #if epoch % 50 == 0:
    print(f"Epoch {epoch}, Training Loss: {avg_loss}, Validation Loss: {avg_val_loss}")

    # Early stopping logic
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

# Testing phase
test_loss = 0.0
model.eval()
with torch.no_grad():
    for batch_inputs, batch_labels in val_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)
        outputs = model(batch_inputs, edge_index,edge_weight )  # Provide the adjacency matrix here
        
        loss = criterion(outputs, batch_labels)
        test_loss += loss.item()

# Average test loss
test_loss /= len(val_loader)
testLossHist.append(test_loss)
print(f"Test Loss: {test_loss}")


