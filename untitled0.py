import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
# Load your data
data = pd.read_csv("data.csv", parse_dates=["datetime"])
labels = pd.read_csv("label.csv", parse_dates=["datetime"])
data_dir = ""
distances = pd.read_csv(
    os.path.join(data_dir, "weights.csv"), header=None
).to_numpy()

distances = torch.tensor(distances)
# Convert datetime strings to actual datetime objects
data['datetime'] = pd.to_datetime(data['datetime'])
labels['datetime'] = pd.to_datetime(labels['datetime'])


# Extract features and labels
features = torch.tensor(data.drop(['datetime'], axis=1).values)  # Assuming 'datetime' is not a feature
labels = torch.tensor(labels.drop(['datetime'], axis=1).values)

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


temporal_horizon = 5
num_samples, num_features = features.shape
X = []
y = []

for i in range(num_samples - temporal_horizon + 1):
    X.append(features[i:i + temporal_horizon].reshape(-1))
    y.append(labels[i + temporal_horizon - 1])

X = torch.stack(X)
y = torch.stack(y)


# Split the data into training and validation sets
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X, y, test_size=0.2, random_state=42)




class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        # Add sigmoid for binary classification
        return torch.sigmoid(x)

# Create the GCN model
model = GCN(input_dim=19*temporal_horizon, hidden_dim=64, output_dim=19)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    output = model(X_train_tensor, edge_index)
    loss = criterion(output, y_train_tensor)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# Evaluation on the test set
with torch.no_grad():
    model.eval()
    test_output = model(X_test_tensor, edge_index)
    predicted_labels = (torch.sigmoid(test_output) > 0.5).float()

# Calculate accuracy on the test set
accuracy = accuracy_score(y_test_tensor.numpy(), predicted_labels.numpy())
print(f'Test Accuracy: {accuracy * 100:.2f}%')
