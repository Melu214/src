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
from sklearn.preprocessing import StandardScaler
#n_sensor = 19

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from scipy.stats import pearsonr
'''
def calculate_autocorrelation_adjacency(file_path, lag=5, correlation_threshold=0.1):
    df = pd.read_csv(file_path, parse_dates=['datetime'], index_col='datetime')

    # Calculate autocorrelation matrix
    num_sensors = len(df.columns)

    adjacency_matrix = np.zeros((num_sensors, num_sensors))

    for i in range(num_sensors):
        for j in range(i + 1, num_sensors):
            correlation_positive_lag = pearsonr(df.iloc[lag:, i], df.iloc[:-lag, j])[0]
            correlation_negative_lag = pearsonr(df.iloc[:-lag, i], df.iloc[lag:, j])[0]
            avg_correlation = (abs(correlation_positive_lag) + abs(correlation_negative_lag)) / 2

            # If average correlation is less than the threshold, set weight to 0, otherwise set weight to the correlation value
            adjacency_matrix[i, j] = 0 if avg_correlation < correlation_threshold else avg_correlation
            adjacency_matrix[j, i] = adjacency_matrix[i, j]  # Since the graph is undirected

    return adjacency_matrix
edge_weight = calculate_autocorrelation_adjacency(file_path)
'''

# Example usage:
file_path = 'data_df1.csv'

data_dir = ""
distances = pd.read_csv(os.path.join(data_dir, "weights.csv"), header=None).to_numpy()








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

def processData(sensor_data):
    #sensor_data = sensor_data.iloc[:,:n_sensor+1]
    #sensor_data_values = sensor_data
    
    return scaler.fit_transform(sensor_data)
   

# Load your data
data = pd.read_csv("data_df1.csv", parse_dates=["datetime"])

labels = pd.read_csv("label_df1.csv", parse_dates=["datetime"])
data_dir = ""
distances = pd.read_csv(os.path.join(data_dir, "weights1.csv"), header=None).to_numpy()
distances[distances > 500] = 0
distances = torch.tensor(distances)
#edge_weight = 1.0 / torch.where(distances == 0, torch.ones_like(distances), distances)
edge_weight = torch.where(distances == 0, torch.zeros_like(distances), 1.0 / distances)
adj_matrix = torch.tensor(edge_weight)
#
# Convert datetime strings to actual datetime objects
data['datetime'] = pd.to_datetime(data['datetime'])
labels['datetime'] = pd.to_datetime(labels['datetime'])

# Extract features and labels
scaler = StandardScaler()
features = torch.tensor(processData(data.drop(['datetime'], axis=1).values))  # Assuming 'datetime' is not a feature

#features = scaler.fit_transform(features)
labels = torch.tensor(labels.drop(['datetime'], axis=1).values, dtype=torch.float32)



# Calculate edge weights as the reciprocal of distances (avoiding division by zero)
#edge_weights = weights.copy()
# Create adjacency matrix with edge weights
#adj_matrix = distances.clone()
#adj_matrix -= torch.diag_embed(torch.diagonal(adj_matrix))

# Create a sparse adjacency matrix in COO format
edge_index = torch.nonzero(adj_matrix).t()
edge_weight = edge_weight[edge_index[0], edge_index[1]]
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

'''

class ComplexGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(ComplexGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        self.attention1 = nn.MultiheadAttention(hidden_dim, 4)
        self.attention2 = nn.MultiheadAttention(hidden_dim, 4)
        self.attention3 = nn.MultiheadAttention(output_dim, 19)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()

    def forward(self, node_features, edge_index, edge_weight):
        # GCN layers with attention
        x = self.conv1(node_features.to(torch.float32), edge_index, edge_weight=edge_weight.to(torch.float32))
        
        x = torch.relu(x)
        x, _ = self.attention1(x, x, x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        
        x = torch.relu(x)
        x, _ = self.attention2(x, x, x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        x, _ = self.attention3(x, x, x)
        x = torch.relu(x)
        
        return self.sigmoid(x)

'''
'''
# Define the ComplexGCN model with layer normalization
class ComplexGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5, weight_decay=0.001):
        super(ComplexGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)
        self.attention1 = nn.MultiheadAttention(hidden_dim, 4)
        self.attention2 = nn.MultiheadAttention(hidden_dim, 4)
        self.attention3 = nn.MultiheadAttention(hidden_dim, 4)
        #self.attention4 = nn.MultiheadAttention(output_dim, 19)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.weight_decay = weight_decay

    def forward(self, node_features, edge_index, edge_weight):
        # GCN layers with attention and layer normalization
        x = self.conv1(node_features.to(torch.float32), edge_index, edge_weight=edge_weight.to(torch.float32))
        x = torch.relu(x)
        x, _ = self.attention1(x, x, x)
        x = self.norm1(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        x = torch.relu(x)
        x, _ = self.attention2(x, x, x)
        x = self.norm2(x)
        x = self.dropout(x)
        
        
        x = self.conv3(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        x = torch.relu(x)
        x, _ = self.attention3(x, x, x)
        x = self.norm3(x)
        
        x = self.conv4(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        x = torch.relu(x)
        x = self.norm4(x)
        
        return self.sigmoid(x)
    
        
    
    def l2_regularization_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)  # L2 norm of parameters
        return self.weight_decay * l2_loss


 
# Move edge_index and edge_weight to the GPU
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)
# Create DataLoader for training data

# Split the data into training and validation sets
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(features, labels, test_size=0.05, random_state=42)



batch_size = 64

# Assuming features and labels are your data tensors
#num_samples = len(features)
#train_size = int(0.8* num_samples)
#val_size = num_samples - train_size

# Use slicing to get the training and validation sets
#X_train_tensor, X_test_tensor = features[:train_size], features[train_size:]
#y_train_tensor, y_test_tensor = labels[:train_size], labels[train_size:]

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


  
# Create the DynamicGCN model
model = ComplexGCN(input_dim=19, hidden_dim=32, output_dim=19)

# Move the entire model to GPU
model = model.to(device)


# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Automatically applies sigmoid internally
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)

# Training loop with early stopping
num_epochs = 1000
best_val_loss = float('inf')
patience = 200
counter = 0

lossHist = []
valLossHist = []
testLossHist = []

for epoch in range(num_epochs):
    # Training
    model.train()
    total_loss = 0.0
    for batch_inputs, batch_labels in train_loader:
        output = model(batch_inputs.to(device), edge_index, edge_weight=edge_weight)
        loss = criterion(output, batch_labels.to(device))
        l2_loss = model.l2_regularization_loss()
        total_loss = loss + l2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_inputs.size(0)
    
    avg_loss = total_loss / len(train_dataset)
    lossHist.append(avg_loss)

    # Validation
    model.eval()
    
    val_loss = 0.0
    with torch.no_grad():
        for batch_inputs, batch_labels in val_loader:
            output = model(batch_inputs.to(device), edge_index, edge_weight=edge_weight)
            loss = criterion(output, batch_labels.to(device))
            val_loss += loss.item() * batch_inputs.size(0)

    avg_val_loss = val_loss / len(val_dataset)
    valLossHist.append(avg_val_loss)

    #val_loss /= len(val_loader)
    scheduler.step(avg_val_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss}, Validation Loss: {avg_val_loss}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

    # Print loss every 10 epochs
    #if (epoch + 1) % 10 == 0:
    #print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
'''
import torch


# Assuming you have initialized 'device' earlier
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the ComplexGCN model with layer normalization
class ComplexGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5, weight_decay=0.001):
        super(ComplexGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, output_dim)
        self.attention1 = nn.MultiheadAttention(hidden_dim, 4)
        self.attention2 = nn.MultiheadAttention(hidden_dim, 4)
        self.attention3 = nn.MultiheadAttention(hidden_dim, 4)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.norm4 = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.weight_decay = weight_decay

    def forward(self, node_features, edge_index, edge_weight):
        # GCN layers with attention and layer normalization
        x = self.conv1(node_features.to(torch.float32), edge_index, edge_weight=edge_weight.to(torch.float32))
        x = torch.relu(x)
        x, _ = self.attention1(x, x, x)
        x = self.norm1(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        x = torch.relu(x)
        x, _ = self.attention2(x, x, x)
        x = self.norm2(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        x = torch.relu(x)
        x, _ = self.attention3(x, x, x)
        x = self.norm3(x)
        
        x = self.conv4(x, edge_index, edge_weight=edge_weight.to(torch.float32))
        x = torch.relu(x)
        x = self.norm4(x)
        
        return self.sigmoid(x)
    
    def l2_regularization_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)  # L2 norm of parameters
        return self.weight_decay * l2_loss


# Move edge_index and edge_weight to the GPU
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)

# Create DataLoader for training data
batch_size = 32

# Split the data into training and validation sets
X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create TensorDatasets and DataLoaders for training and validation
#train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_dataset = TensorDataset(features, labels)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Create the ComplexGCN model
model = ComplexGCN(input_dim=76, hidden_dim=128, output_dim=19).to(device)

# Define loss and optimizer
criterion = nn.BCEWithLogitsLoss()  # Automatically applies sigmoid internally
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)

# Training loop with early stopping
num_epochs = 10000
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
    for batch_inputs, batch_labels in train_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        output = model(batch_inputs, edge_index, edge_weight=edge_weight)
        loss = criterion(output, batch_labels)
        l2_loss = model.l2_regularization_loss()
        total_loss = loss + l2_loss
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
    scheduler.step(avg_val_loss)
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





# Test the model on the test set if needed
test_loss = 0.0
model.eval()
with torch.no_grad():
    test_output = model(X_val_tensor.to(device), edge_index, edge_weight=edge_weight)
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



'''



# Training loop
num_epochs = 1000
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
'''