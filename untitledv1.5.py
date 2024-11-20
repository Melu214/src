import torch
from torch_geometric.nn import GraphConv
from torch_geometric.data import Data
import pandas as pd



##


def build_edge_index(distance_matrix):
  """
  Creates edge_index for a graph based on a distance threshold.

  Args:
    distance_matrix: A torch tensor of size (n_sensors x n_sensors) containing pairwise distances.

  Returns:
    edge_index: A torch tensor of size (2 x n_edges) containing source and target sensor indices for each edge.
  """
  # Compare distances with the threshold
  adj_matrix = torch.where(distance_matrix < 700, torch.ones_like(distance_matrix), torch.zeros_like(distance_matrix))

  # Remove self-loops and diagonal entries
  adj_matrix = adj_matrix - torch.eye(adj_matrix.size(0), dtype=torch.bool)

  # Create edge_index from the adjacency matrix
  edge_index = torch.nonzero(adj_matrix, dim=2)

  # Return shuffled edge indices to avoid bias
  return edge_index.t().permute(1, 0)  # Transpose and swap dimensions for correct format


# Define sensor data class
class SensorData(Data):
    def __init__(self, sensor_features, labels, distance_matrix, **kwargs):
        super().__init__(**kwargs)
        self.sensor_features = sensor_features
        self.labels = labels
        self.distance_matrix = distance_matrix
        
#read data

data = pd.read_csv("data_df1.csv", parse_dates=["datetime"])

labels = pd.read_csv("label_df1.csv", parse_dates=["datetime"])
# Convert datetime strings to actual datetime objects
data['datetime'] = pd.to_datetime(data['datetime'])
labels['datetime'] = pd.to_datetime(labels['datetime'])
data = data.drop(['datetime'], axis=1).values
labels = labels.drop(['datetime'], axis=1).values
# Extract features and labels


# Load your data and prepare training...
sensor_features = torch.tensor(data, dtype=torch.float)  # Replace with your data
labels = torch.tensor(labels, dtype=torch.long)  # Replace with your labels
distance = pd.read_csv('weights.csv', header=None).to_numpy()
distance_matrix = torch.tensor(distance, dtype=torch.float)  # Replace with your distances
edge_index = torch.tensor(build_edge_index(distance_matrix))  # Function to create edges based on distance threshold

# Build training and validation datasets
train_data = SensorData(sensor_features[:], labels[:], distance_matrix)
val_data = SensorData(sensor_features[::2], labels[::2], distance_matrix)

# Define GNN model
class GNNModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.conv3 = GraphConv(hidden_dim, out_dim)

    def forward(self, data):
        x = data.sensor_features
        edge_index = data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.nn.functional.relu(x)
        x = self.conv3(x, edge_index)
        return x

# Define loss function and optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):  # Adjust number of epochs
    model.train()
    optimizer.zero_grad()
    pred = model(train_data)
    loss = loss_fn(pred, train_data.labels)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
val_pred = model(val_data)
val_loss = loss_fn(val_pred, val_data.labels)
accuracy = torch.sum(val_pred.argmax(dim=1) == val_data.labels).item() / len(val_data.labels)
precision = torch_geometric.utils.dense_adj(val_data.edge_index)[val_pred.argmax(dim=1) == val_data.labels].mean()
recall = torch_geometric.utils.dense_adj(val_data.edge_index)[val_data.labels == val_pred.argmax(dim=1)].mean()
auc = torch_geometric.utils.auc(val_pred, val_data.labels)

print(f"Epoch: {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")

