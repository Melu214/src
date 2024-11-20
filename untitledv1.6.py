import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# Assuming you have initialized 'device' earlier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load your sensor data and labels
data = pd.read_csv("data_df1.csv", parse_dates=["datetime"])

labels = pd.read_csv("label_df1.csv", parse_dates=["datetime"])
# Convert datetime strings to actual datetime objects
data['datetime'] = pd.to_datetime(data['datetime'])
labels['datetime'] = pd.to_datetime(labels['datetime'])
data = data.drop(['datetime'], axis=1).values
labels = labels.drop(['datetime'], axis=1).values

# Assuming you have a distance matrix 'distance_matrix'
# Replace 'your_distance_matrix.csv' with your actual file name
import numpy as np

# Identify rows where all column values are 0
zero_rows = np.all(labels == 0, axis=1)

# Randomly select 50% of these rows to delete
rows_to_delete = np.random.choice(np.where(zero_rows)[0], size=int(0.9 * np.sum(zero_rows)), replace=False)

# Delete selected rows from both labels and data
labels = np.delete(labels, rows_to_delete, axis=0)
data = np.delete(data, rows_to_delete, axis=0)



# Convert data to PyTorch tensors
sensor_data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)



distances = pd.read_csv("weights1.csv", header=None).to_numpy()
distances[distances > 500] = 0
distances = torch.tensor(distances)
#edge_weight = 1.0 / torch.where(distances == 0, torch.ones_like(distances), distances)
edge_weight = torch.where(distances == 0, torch.zeros_like(distances), 1.0 / distances)
adj_matrix = edge_weight.clone()


# Create a sparse adjacency matrix in COO format
edge_index = torch.nonzero(adj_matrix).t()
edge_weight = edge_weight[edge_index[0], edge_index[1]]
adj_matrix_sparse = torch.sparse_coo_tensor(
    edge_index, edge_weight, torch.Size([sensor_data.shape[0], sensor_data.shape[0]])
)

# Convert to COO format
edge_index = adj_matrix_sparse.coalesce().indices()
edge_weight = adj_matrix_sparse.coalesce().values()





# Define your GNN model
class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight):
        x = self.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
        x = self.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
        x = self.conv3(x, edge_index, edge_weight=edge_weight)
        x = self.sigmoid(x)
        return x

# Move data and model to GPU if available
sensor_data, labels, edge_weight, edge_index = sensor_data.to(device), labels.to(device), edge_weight.to(device), edge_index.to(device)
model = SimpleGCN(input_dim=76, hidden_dim=32, output_dim=19).to(device)

# Assuming you have split your data into training and validation sets
X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(sensor_data, labels, test_size=0.05, random_state=42)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy with Logits
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for batch_inputs, batch_labels in train_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        output = model(batch_inputs, edge_index, edge_weight=edge_weight)
        loss = criterion(output, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_inputs.size(0)

    avg_loss = total_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}')

# Validation
model.eval()
val_loss = 0.0
with torch.no_grad():
    for batch_inputs, batch_labels in val_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        output = model(batch_inputs, edge_index, edge_weight=edge_weight)
        loss = criterion(output, batch_labels)
        val_loss += loss.item() * batch_inputs.size(0)

avg_val_loss = val_loss / len(val_loader.dataset)
print(f'Validation Loss: {avg_val_loss:.4f}')
