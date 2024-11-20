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

data1 = data.copy()

data = data1[:,0:4]

import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# Assuming 'sensor_data' is your time series data
# Flatten the time series data into a 1D array
data = data.flatten().reshape(-1, 1)

# Train Isolation Forest
model = IsolationForest(contamination=0.05)  # Adjust contamination based on your dataset
model.fit(data)

# Predict anomalies
anomaly_scores = model.decision_function(data)
labelv1 = model.predict(data)

# Visualize anomalies
plt.figure(figsize=(10, 6))
plt.plot(data.flatten(), label='Sensor Data')
plt.scatter(np.where(labelv1 == -1), data.flatten()[labelv1 == -1], color='red', label='Anomalies')
plt.title('Anomaly Detection using Isolation Forest')
plt.legend()
plt.show()




import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Assuming 'data' is your 2D array with 4 features
# 'data' should be a 2D numpy array (num_samples, num_features)

# Set parameters
sequence_length = 20
hidden_size = 16
threshold = 0.1
num_epochs = 50
batch_size = 32

# Function to create sequences from the data
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i : i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

# Convert data to PyTorch tensor
data = torch.tensor(data, dtype=torch.float32)

# Create sequences
sequences = create_sequences(data, sequence_length)

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

# Move data and model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sequences = sequences.to(device)

# Initialize and train the autoencoder
input_size = sequences.size(-1)  # Number of features
autoencoder = Autoencoder(input_size, hidden_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)

# Create a DataLoader for training
train_dataset = TensorDataset(sequences)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(20):
    for batch_inputs in train_loader:
        batch_inputs = batch_inputs[0].to(device)

        optimizer.zero_grad()
        outputs = autoencoder(batch_inputs)
        loss = criterion(outputs, batch_inputs)
        loss.backward()
        optimizer.step()

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

# Set a threshold for anomaly detection
anomalies = [1 if error > 500 else 0 for error in reconstruction_errors]
print(anomalies)



from sklearn.metrics import confusion_matrix

conf_matrix = confusion_matrix(y_true, y_pred_binary)
print('Confusion Matrix:')
print(conf_matrix)

# Assuming you have a distance matrix 'distance_matrix'
# Replace 'your_distance_matrix.csv' with your actual file name

# Identify rows where all column values are 0
zero_rows = np.all(labels == 0, axis=1)
num_zero_rows = np.sum(zero_rows)

num_rows_to_eliminate = int(0.75 * num_zero_rows)


# Get the indices of zero rows to eliminate
rows_to_eliminate = np.where(zero_rows)[0]
rows_to_eliminate = np.random.choice(rows_to_eliminate, size=num_rows_to_eliminate, replace=False)

# Eliminate the selected rows from 'labels'
labels = np.delete(labels, rows_to_eliminate, axis=0)
data = np.delete(data,rows_to_eliminate, axis=0)



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



class ComplexGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ComplexGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, edge_weight):
        x = self.relu(self.conv1(x.to(torch.float32), edge_index, edge_weight=edge_weight.to(torch.float32)))
        x = self.relu(self.conv2(x.to(torch.float32), edge_index, edge_weight=edge_weight.to(torch.float32)))
        x = self.conv3(x.to(torch.float32), edge_index, edge_weight=edge_weight.to(torch.float32))
        x = self.sigmoid(x)
        return x
    




# Move data and model to GPU if available
sensor_data, labels, edge_weight, edge_index = sensor_data.to(device), labels.to(device), edge_weight.to(device), edge_index.to(device)
#model = ComplexGCN(input_dim=76, hidden_dim=32, output_dim=19).to(device)

# Assuming you have split your data into training and validation sets
X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(sensor_data, labels, test_size=0.2, random_state=42)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


model = ComplexGCN(input_dim=76, hidden_dim=64, output_dim=19).to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Automatically applies sigmoid internally
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)

# Training loop with early stopping
num_epochs = 15
best_val_loss = float('inf')
patience = 200  # Adjusted patience
counter = 0


lossHist = []
valLossHist = []
testLossHist= []
for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0.0
    for i, (batch_inputs, batch_labels) in enumerate(train_loader):
        #print(i)
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        output = model(batch_inputs, edge_index, edge_weight=edge_weight)
        loss = criterion(output, batch_labels)
        #l2_loss = model.l2_regularization_loss()
        total_loss = loss #+ l2_loss
        total_loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_inputs.size(0)

    avg_loss = total_loss / len(train_dataset)
    lossHist.append(avg_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            output = model(batch_inputs, edge_index, edge_weight=edge_weight)
            loss = criterion(output, batch_labels)
            val_loss += loss.item() * batch_inputs.size(0)

    avg_val_loss = val_loss / len(val_dataset)
    valLossHist.append(avg_val_loss)

    # Learning rate scheduler and printing
    #scheduler.step(avg_val_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

torch.save(model.state_dict(), 'best_model1.pth')


# Load the best model after training
best_model = ComplexGCN(input_dim=76, hidden_dim=64, output_dim=19).to(device)
best_model.load_state_dict(torch.load('best_model1.pth'))
best_model.eval()

from sklearn.metrics import confusion_matrix



# Evaluate on test set
# Evaluate on test set
test_loss = 0.0
y_true = []
y_pred = []

with torch.no_grad():
    for batch_inputs, batch_labels in train_loader:  # Assuming you have a test_loader
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        output = best_model(batch_inputs, edge_index, edge_weight=edge_weight)
        loss = criterion(output, batch_labels)
        test_loss += loss.item() * batch_inputs.size(0)

        y_true.extend(batch_labels.cpu().numpy())
        y_pred.extend(output.detach().cpu().numpy())  # Use detach() to avoid the RuntimeError

avg_test_loss = test_loss / len(train_loader.dataset)
testLossHist.append(avg_test_loss)

# Convert the probabilities to binary predictions
y_pred_binary = np.round(np.array(y_pred))


# Compute the confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred_binary)
print('Confusion Matrix:')
print(conf_matrix)




best_model = ComplexGCN(input_dim=76, hidden_dim=64, output_dim=19).to(device)
best_model.load_state_dict(torch.load('best_model.pth'))
best_model.eval()
# Test the model on the test set if needed
test_loss = 0.0
best_model.eval()
with torch.no_grad():
    test_output = best_model(X_val_tensor.to(device), edge_index, edge_weight=edge_weight)
    loss = criterion(test_output, y_val_tensor.to(device))
    test_loss += loss.item()
# Average test loss
test_loss /= len(val_loader)
testLossHist.append(test_loss)
print(f"Test Loss: {test_loss}")
#print(f'Test Loss: {test_loss.item():.4f}')






from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set the global font to be Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14  # Base font size
mpl.rcParams['axes.titlepad'] = 20 
mpl.rcParams['axes.labelpad'] = 10 

# Assuming you've executed all the necessary parts above related to torch and data processing

# Flattening the true labels from the validation set
y_true_val_flat = [label for sample in val_dataset for label in sample[1]]

# Getting the flat list of predicted scores
y_scores_val_flat = []

model.eval()
with torch.no_grad():
    for batch_inputs, _ in val_loader:
        batch_inputs = batch_inputs.to(device)
        outputs = model(batch_inputs.to(device), edge_index, edge_weight=edge_weight)
        for output in outputs:
            y_scores_val_flat.extend(torch.sigmoid(output).cpu().numpy())

# Calculate precision and recall values
precision, recall, _ = precision_recall_curve(y_true_val_flat, y_scores_val_flat)

# Plotting with adjustments
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', linestyle='-')

# Titles, labels, and other adjustments
plt.title('Precision-Recall Curve for All Sensors Combined', fontsize=18, fontname="Times New Roman")
plt.xlabel('Recall', fontsize=16, fontname="Times New Roman")
plt.ylabel('Precision', fontsize=16, fontname="Times New Roman")

# Ticks
plt.xticks(fontsize=14, fontname="Times New Roman")
plt.yticks(fontsize=14, fontname="Times New Roman")

plt.grid(True)

# Saving the plot with high resolution
plt.tight_layout()
plt.savefig('Precision_Recall_Curve_high_res.png', dpi=300)

# Show the plot
plt.show()







from sklearn.metrics import f1_score, roc_auc_score, roc_curve
import matplotlib as mpl

# Setting the global font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14  # Base font size
mpl.rcParams['axes.titlepad'] = 20 
mpl.rcParams['axes.labelpad'] = 10 

# Assuming you've executed the necessary parts above related to torch and data processing

model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_inputs, batch_labels in val_loader:
        #adjacency_tensor = adjacency_tensor.to(device)
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        outputs = model(batch_inputs.to(device), edge_index, edge_weight=edge_weight)
        # Assuming you're doing a classification task & using a sigmoid activation in the last layer
        predictions = torch.sigmoid(outputs).data.cpu().numpy()
        all_preds.extend(predictions)
        all_true.extend(batch_labels.data.cpu().numpy())

# Calculate F1-Score
all_preds = np.array(all_preds)
all_true = np.array(all_true)
all_true_1d = all_true.ravel()
all_preds_1d = (all_preds > 0.5).astype(int).ravel()
f_score = f1_score(all_true_1d, all_preds_1d)
print(f"F1-Score: {f_score}")

# Calculate AUC
auc_score = roc_auc_score(all_true_1d, all_preds_1d)
print(f"AUC: {auc_score}")



#

fpr, tpr, _ = roc_curve(all_true_1d, all_preds_1d)
# Plotting the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

# Titles, labels, and other adjustments
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18, fontname="Times New Roman")
plt.xlabel('False Positive Rate', fontsize=16, fontname="Times New Roman")
plt.ylabel('True Positive Rate', fontsize=16, fontname="Times New Roman")

# Ticks
plt.xticks(fontsize=14, fontname="Times New Roman")
plt.yticks(fontsize=14, fontname="Times New Roman")

plt.legend(loc="lower right", fontsize=14, frameon=True, edgecolor='black')

# Saving the plot with high resolution
plt.tight_layout()
plt.savefig('ROC_Curve_high_res.png', dpi=300)

# Display the plot
plt.show()




##############
##############

#CONFUSION MATRIX
###############
####################
#pip install kaleido
#pip install plotly



import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
import numpy as np

# Assuming that you've already calculated `all_preds_1d` and `all_true_1d` from the previous steps

# Generating the confusion matrix
cm = confusion_matrix(all_true_1d, all_preds_1d)

# Convert the matrix into DataFrame for labeling purpose
labels = ['Negative', 'Positive']  # or whatever is appropriate for your binary classification task
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Create a heatmap with plotly
fig = ff.create_annotated_heatmap(
    z=cm,
    x=list(cm_df.columns),
    y=list(cm_df.index),
    annotation_text=cm_df.values,
    colorscale='Viridis'
)

# Modify the layout
fig.update_layout(
    title='Confusion Matrix',
    title_font=dict(family="Times New Roman", size=22),
    xaxis=dict(title='Predicted Label', title_font=dict(family="Times New Roman", size=18),
               tickfont=dict(family="Times New Roman", size=14)),
    yaxis=dict(title='True Label', title_font=dict(family="Times New Roman", size=18),
               tickfont=dict(family="Times New Roman", size=14)),
    autosize=False,
    width=800,
    height=600,
)

# Show the figure
fig.show()

# To save the figure in a high resolution suitable for a paper:
fig.write_image("confusion_matrix_high_res.png", scale=3)


############
############
#ROC  CURVE#
############
############






from sklearn.metrics import f1_score, roc_auc_score, roc_curve


# After testing phase, calculate predictions for the test set
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_inputs, batch_labels in val_loader:
        #adjacency_tensor = adjacency_tensor.to(device)
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        outputs = model(batch_inputs.to(device), edge_index, edge_weight=edge_weight)
        # Assuming you're doing a classification task & using a sigmoid activation in the last layer
        predictions = torch.sigmoid(outputs).data.cpu().numpy()
        all_preds.extend(predictions)
        all_true.extend(batch_labels.data.cpu().numpy())

# Calculate F1-Score
all_preds = np.array(all_preds)
all_true = np.array(all_true)
all_true_1d = all_true.ravel()
all_preds_1d = (all_preds > 0.5).astype(int).ravel()
f_score = f1_score(all_true_1d, all_preds_1d)
 # Using 0.5 as the threshold
 # Using 0.5 as the threshold
print(f"F1-Score: {f_score}")

# Calculate AUC
auc_score = roc_auc_score(all_true_1d, all_preds_1d)
print(f"AUC: {auc_score}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(all_true_1d, all_preds_1d)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

