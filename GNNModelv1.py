# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 17:03:52 2023

@author: Usama
"""

import pandas as pd
import os
from fancyimpute import IterativeImputer
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

'''
def labeller(df):
    df1 = df.copy()
    df1.set_index('DateTimeStamp', inplace=True)
    # Add Hurricane_Label column
    df1['Hurricane_Label'] = 0

    # Set the threshold values for turbidity rise and fall
    rise_threshold = 150  # Lower threshold for turbidity rise

    # Set the minimum duration for a sustained rise (you can adjust this based on your data)
    min_duration = 4  # Minimum duration in hours

    # Initialize variables to track the start and end of a potential hurricane event
    hurricane_start = None

    # Iterate through the DataFrame to identify and label hurricane-affected data
    for index, row in df1.iterrows():
        turbidity = row[df1.columns[-1]]

        # Check if a potential hurricane event is starting
        if turbidity > rise_threshold and hurricane_start is None:
            hurricane_start = index

        # Check if a potential hurricane event is ending
        elif turbidity < rise_threshold and hurricane_start is not None:
            if (index - hurricane_start).seconds / 3600 < min_duration:
                hurricane_start = None
            else:
                hurricane_end = index
                actual_start = hurricane_start - pd.Timedelta(hours=15)
                actual_end = hurricane_end + pd.Timedelta(hours=15)
                # Label the data points within the hurricane event as 1
                df1.loc[actual_start:actual_end, 'Hurricane_Label'] = 1

                # Reset the hurricane_start variable
                hurricane_start = None

    print(df1['Hurricane_Label'].describe())
    return df1
'''

# Set the directory path to your CSV files
folder_path = 'C:/Users/mi0025/Documents/Research/anomalydetector/CSVData/'

# Define the date range
start_date = '2007-01-01'
end_date = '2023-09-30'

# Create an empty DataFrame to store the combined data
combined_df = pd.DataFrame()
label_df = pd.DataFrame()

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)
        
       
        
        # Read the CSV file
        df = pd.read_csv(file_path, skiprows=2)
        df = df[['DateTimeStamp', 'Temp', 'Sal', 'DO_mgl', 'Turb']]
        df['DateTimeStamp'] = pd.to_datetime(df['DateTimeStamp'], format='%m/%d/%Y %H:%M')

        # Filter the DataFrame to include data within the specified date range
        df = df[(df['DateTimeStamp'] >= start_date) & (df['DateTimeStamp'] <= end_date)]
        df.set_index('DateTimeStamp', inplace=True)
        df = df.resample('2H').mean()
        
        # Perform interpolation using fancyimpute
        if len(df)!= 587137:
            merged_df = pd.DataFrame(index=combined_df.index)
            df = pd.concat([merged_df, df], axis=1)
        cols = df.columns
        for col in cols:
            df[col] = df[col].interpolate()
            imputer = IterativeImputer()
            df[col] = imputer.fit_transform(df[col].values.reshape(-1, 1))
            
            

            
        # Add filename as a prefix to each column
        df.columns = [f"{filename.split('.')[0]}_{col}" if col != 'DateTimeStamp' else col for col in df.columns]
        
        #principle componenets to transpform multivariate to univariate
        pca = PCA(n_components=1)  # Set the number of components as needed
        #scaler = MinMaxScaler()
        latent_variable = pca.fit_transform(df)
        df[f"{filename.split('.')[0]}"] = latent_variable
        df_r = df[f"{filename.split('.')[0]}"]
        
        combined_df = pd.concat([combined_df, df_r], axis=1)
                       
        # Create a column 'Hurricane_Label' and initialize it with 0 (normal data)
        df['Hurricane_Label'] = 0
        
        # Set the threshold values for turbidity rise and fall
        rise_threshold = 150  # Lower threshold for turbidity rise
        
        # Set the minimum duration for a sustained rise (you can adjust this based on your data)
        min_duration = 4  # Minimum duration in hours
        
        # Initialize variables to track the start and end of a potential hurricane event
        hurricane_start = None
        
        # Iterate through the DataFrame to identify and label hurricane-affected data
        for index, row in df.iterrows():
            turbidity = row[f"{filename.split('.')[0]}_{col}"]
        
            # Check if a potential hurricane event is starting
            if turbidity > rise_threshold and hurricane_start is None:
                hurricane_start = index
        
            # Check if a potential hurricane event is ending
            elif (turbidity < rise_threshold and hurricane_start is not None ):
                if((index - hurricane_start).seconds / 3600 < min_duration):
                    hurricane_start = None
                else:
                    hurricane_end = index
                    actual_start = hurricane_start - pd.Timedelta(hours=10)
                    actual_end = hurricane_end + pd.Timedelta(hours=10)
                    # Label the data points within the hurricane event as 1
                    df.loc[actual_start:actual_end, 'Hurricane_Label'] = 1
            
                    # Reset the hurricane_start variable
                    hurricane_start = None
                
        label_df[f"{filename.split('.')[0]}"] = df['Hurricane_Label']
        del df
        del df_r
      

label_df.to_csv('label_dfpca.csv', index=True)
combined_df.to_csv('data_dfpca.csv', index=True)


import pandas as pd
from geopy.distance import geodesic
import numpy as np
import openpyxl
# Read the Excel file into a DataFrame
file_path = 'latlons.xlsx'  # Replace with the actual file path
df = pd.read_excel(file_path)

def compute_adjacency_matrix(
    route_distances: np.ndarray, sigma2: float, epsilon: float
):
    """Computes the adjacency matrix from distances matrix.

    It uses the formula in https://github.com/VeritasYin/STGCN_IJCAI-18#data-preprocessing to
    compute an adjacency matrix from the distance matrix.
    The implementation follows that paper.

    Args: num_routes)`. Entry `i,j` of this array is the
            distance between roads `i,j`.
        sigma2: Determines the width of the Gaussian kernel applied to the square distances matrix.
        epsilon: A threshold specifying if there is an edge between two nodes. Specifically, `A[i,j]=1`
            if `np.exp(-w2[i,j] / sigma2) >= epsilon` and `A[i,j]=0` otherwise, where `A` is the adjacency
            matrix and `w2=route_distances * route_distances`

    Returns:
        A boolean graph adjacency matrix.
    """
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 100000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask





# Function to calculate distance between two points
def calculate_distance(coords1, coords2):
    return geodesic(coords1, coords2).kilometers

# Generate a distance matrix
n = len(df)
distance_matrix = pd.DataFrame(index=df.index, columns=df.index)

for i in range(n):
    for j in range(i, n):
        coords1 = (df.at[i, 'latitude'], df.at[i, 'longitude'])
        coords2 = (df.at[j, 'latitude'], df.at[j, 'longitude'])
        distance_matrix.at[i, j] = calculate_distance(coords1, coords2)
        distance_matrix.at[j, i] = distance_matrix.at[i, j]

# Print or use the distance matrix as needed
print(distance_matrix)


import numpy as np
#route_distances = np.array(distance_matrix)
data_dir = ""
distance_matrix.to_csv("weights1.csv", header = False, index = False)
route_distances = pd.read_csv(
    os.path.join(data_dir, "weights.csv"), header=None
).to_numpy()


class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes


sigma2 = 0.1
epsilon = 0.5
adjacency_matrix = compute_adjacency_matrix(route_distances, sigma2, epsilon)
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)

print(adjacency_matrix)

import requests
import zipfile
import os

url = "https://github.com/VeritasYin/STGCN_IJCAI-18/raw/master/dataset/PeMSD7_Full.zip"
data_dir = ""  # specify the target directory where you want to save the data

# Download the zip file
response = requests.get(url)
zip_file_path = os.path.join(data_dir, "PeMSD7_Full.zip")

with open(zip_file_path, "wb") as zip_file:
    zip_file.write(response.content)

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
    zip_ref.extractall(data_dir)

# Remove the zip file if needed
os.remove(zip_file_path)
route_distances = pd.read_csv(
    os.path.join(data_dir, "PeMSD7_W_228.csv"), header=None
).to_numpy()
speeds_array = pd.read_csv(
    os.path.join(data_dir, "PeMSD7_V_228.csv"), header=None
).to_numpy()

print(f"route_distances shape={route_distances.shape}")
print(f"speeds_array shape={speeds_array.shape}")




import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

# Load data from CSV files
sensor_data = pd.read_csv('data_df.csv')
labels_data = pd.read_csv('label_df.csv')
labels_data.drop('Hurricane_Label', axis=1, inplace=True)



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
    sensor_data = sensor_data.iloc[:,:n_sensor+1]
    sensor_data_values = sensor_data.iloc[:, 1:].values
    data_mean = np.mean(sensor_data_values)
    data_std = np.std(sensor_data_values)
    sensor_data_values = (sensor_data_values - data_mean) / data_std
    return sensor_data_values
    


n_sensor = n_sensor = labels_data.shape[1]-1 # n_sensor = 67
# Standardize the data (it's important for neural networks)

sensor_data_values = processData(sensor_data, n_sensor)

# Convert to tensor
adjacency_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)



class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input, adj=None):
        if adj is not None:  # Only multiply by adjacency if provided
            input = torch.matmul(input, adj)
        support = torch.matmul(input, self.weight)
       
        output = support + self.bias
        return output






class GNN(nn.Module):
    def __init__(self, num_nodes, hidden_features, dropout_prob=0.5, weight_decay=0.001):
        super(GNN, self).__init__()
        
        # Input layer
        self.gc1 = GraphConvolution(num_nodes, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)  # Adding batch normalization
        
        # Hidden layer 1
        self.gc2 = GraphConvolution(hidden_features, hidden_features * 2)
        self.bn2 = nn.BatchNorm1d(hidden_features * 2)
        
        # Hidden layer 2
        #self.gc3 = GraphConvolution(hidden_features * 2, hidden_features * 4)
        #self.bn3 = nn.BatchNorm1d(hidden_features * 4)
        
        # Hidden layer 3 (for variety, we'll reduce the number of features here)
        #self.gc4 = GraphConvolution(hidden_features * 4, hidden_features * 2)
        #self.bn4 = nn.BatchNorm1d(hidden_features * 2)
        
        # Output layer
        self.gc5 = GraphConvolution(hidden_features * 2, num_nodes)
        
        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU()
        self.weight_decay = weight_decay

    def forward(self, x, adj):
        # Input layer
        x = self.gc1(x, adj)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Hidden layer 1
        x = self.gc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Hidden layer 2
        #x = self.gc3(x)
        #x = self.bn3(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        
        # Hidden layer 3
        #x = self.gc4(x)
        #x = self.bn4(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        
        # Output layer
        x = self.gc5(x)
        
        return x

    def l2_regularization_loss(self):
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, p=2)  # L2 norm of parameters
        return self.weight_decay * l2_loss





hidden_features = 32

# Model, Loss and Optimizer

model = GNN(num_nodes = n_sensor, hidden_features=hidden_features)
base_learning_rate = 0.01
criterion = nn.BCEWithLogitsLoss()  # As it's a binary classification
optimizer = optim.Adam(model.parameters(), lr=base_learning_rate)
# Early stopping parameters
patience = 10
best_loss = float('inf')
counter = 0

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=patience, verbose=True)
# Assuming 'inputs' and 'labels' are your data tensors
inputs = torch.tensor(sensor_data_values, dtype=torch.float32)
labels = torch.tensor(labels_data.iloc[:, 1:].values, dtype=torch.float32)

# Split data for training and validation
num_samples = len(inputs)
train_size = int(0.9 * num_samples)
val_size = num_samples - train_size

# Create Tensor datasets
train_dataset = TensorDataset(inputs[:train_size], labels[:train_size])
val_dataset = TensorDataset(inputs[train_size:], labels[train_size:])

# Create DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)



model = model.to(device)
adjacency_tensor = adjacency_tensor.to(device)

epochs = 50
lossHist = []
valLossHist = []
testLossHist = []
for epoch in range(epochs):
    model.train()  # Switch to training mode
    total_loss = 0
    
    for batch_inputs, batch_labels in train_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        adjacency_tensor = adjacency_tensor.to(device)
        optimizer.zero_grad()
        outputs = model(batch_inputs, adjacency_tensor)
        loss = criterion(outputs, batch_labels)
        l2_loss = model.l2_regularization_loss()
        total_loss = loss + l2_loss
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
            adjacency_tensor = adjacency_tensor.to(device)
            outputs = model(batch_inputs, adjacency_tensor)
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
        adjacency_tensor = adjacency_tensor.to(device)
        outputs = model(batch_inputs, adjacency_tensor)  # Provide the adjacency matrix here
        
        loss = criterion(outputs, batch_labels)
        test_loss += loss.item()

# Average test loss
test_loss /= len(val_loader)
testLossHist.append(test_loss)
print(f"Test Loss: {test_loss}")

######
######

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
        outputs = model(batch_inputs, adjacency_tensor)
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



####################

##################






######################
#######################
####################
#ROC_Curve_high_
######################


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
        adjacency_tensor = adjacency_tensor.to(device)
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        outputs = model(batch_inputs, adjacency_tensor)
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


#############
###########
#ROC CURVE
#####
#######







from sklearn.metrics import f1_score, roc_auc_score, roc_curve


# After testing phase, calculate predictions for the test set
model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_inputs, batch_labels in val_loader:
        adjacency_tensor = adjacency_tensor.to(device)
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        outputs = model(batch_inputs, adjacency_tensor)
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




########
#########
#OLD Training loss plot
###########
##########
#
#
##
##


import matplotlib.pyplot as plt

# Plot training and validation loss
plt.figure(figsize=(12, 6))

lossHist = [item.cpu().detach().numpy() for item in lossHist]

#lossHist = lossHist.cpu().detach().numpy()


if torch.is_tensor(lossHist):
    lossHist = [item.cpu().detach().numpy() for item in lossHist]
plt.plot(lossHist, label='Training Loss')
plt.plot(valLossHist, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Display test loss
print(f"Test Loss: {test_loss}")

# If you want to visualize the test loss over time
# Uncomment the following:
'''
plt.figure(figsize=(12, 6))
plt.plot(testLossHist, label='Test Loss', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Test Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()

'''

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Assuming you have a way to get the predicted probabilities, y_scores, for the validation set
#y_true_val = [item[1].item() for item in val_dataset]  # True labels for the validation set
y_scores_val = []  # You'll need to populate this list with predicted probabilities

model.eval()
with torch.no_grad():
    for batch_inputs, _ in val_loader:
        batch_inputs = batch_inputs.to(device)
        outputs = model(batch_inputs, adjacency_tensor)
        y_scores_val.extend(torch.sigmoid(outputs).cpu().numpy())

#precision, recall, _ = precision_recall_curve(y_true_val, y_scores_val)




# Flattening the true labels from the validation set
y_true_val_flat = [label for sample in val_dataset for label in sample[1]]

# Getting the flat list of predicted scores
y_scores_val_flat = []

model.eval()
with torch.no_grad():
    for batch_inputs, _ in val_loader:
        batch_inputs = batch_inputs.to(device)
        outputs = model(batch_inputs, adjacency_tensor)
        for output in outputs:
            y_scores_val_flat.extend(torch.sigmoid(output).cpu().numpy())

# Now, let's plot the precision-recall curve
precision, recall, _ = precision_recall_curve(y_true_val_flat, y_scores_val_flat)

plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve for All Sensors Combined')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.grid()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Filter the scores to visualize predictions between 0.4 and 1.0
filtered_scores = [score for score in y_scores_val_flat if 0.4 <= score <= 1.0]

plt.hist(filtered_scores, bins=np.linspace(0.4, 1.0, 20), alpha=0.75)
plt.title('Histogram of Predicted Probabilities between 0.4 and 1.0')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()




import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set the global font to be Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14  # Base font size
mpl.rcParams['axes.titlepad'] = 20 
mpl.rcParams['axes.labelpad'] = 10 

# Your data extraction process (unchanged)
positive_class = [reading for item in dataset for reading, label in zip(item[0].numpy().flatten(), item[1].numpy().flatten()) if label == 1]
negative_class = [reading for item in dataset for reading, label in zip(item[0].numpy().flatten(), item[1].numpy().flatten()) if label == 0]

# Normalize your data to [0, 1]
positive_class = (positive_class - np.min(positive_class)) / (np.max(positive_class) - np.min(positive_class))
negative_class = (negative_class - np.min(negative_class)) / (np.max(negative_class) - np.min(negative_class))

# Plotting with adjustments
plt.figure(figsize=(8, 6))
plt.hist(positive_class, bins=50, alpha=0.5, label='Positive Class', density=True)
plt.hist(negative_class, bins=50, alpha=0.5, label='Negative Class', density=True)

# Titles, labels and other adjustments
plt.title('Distribution of Sensor Speed', fontsize=18, fontname="Times New Roman")
plt.xlabel('Normalized Speed', fontsize=16, fontname="Times New Roman")
plt.ylabel('Density', fontsize=16, fontname="Times New Roman")

# Ticks
plt.xticks(fontsize=14, fontname="Times New Roman")
plt.yticks(fontsize=14, fontname="Times New Roman")

plt.legend(loc="upper right", title="Classes", fontsize=14, title_fontsize=16, frameon=True)

# Saving the plot with high resolution
plt.tight_layout()
plt.savefig('Speed_Distribution_high_res.png', dpi=300)

# Show the plot
plt.show()


'''

'''

positive_class = [reading for item in dataset for reading, label in zip(item[0].numpy().flatten(), item[1].numpy().flatten()) if label == 1]
negative_class = [reading for item in dataset for reading, label in zip(item[0].numpy().flatten(), item[1].numpy().flatten()) if label == 0]

plt.hist(positive_class, bins=50, alpha=0.5, label='Positive Class', density=True)
plt.hist(negative_class, bins=50, alpha=0.5, label='Negative Class', density=True)
plt.title('Distribution of Speed')
plt.xlabel('Sensor Value')
plt.ylabel('Frequency')
plt.legend(loc='upper right')
plt.show()
'''
'''
# You need predictions for this, similar to y_scores_val but thresholded (e.g., > 0.5)
y_pred_val  = [1 if score > 0.5 else 0 for score in y_scores_val_flat]
errors = [i for i, (true, pred) in enumerate(zip(y_true_val_flat, y_pred_val )) if true != pred]

# Display these instances or plot them based on your needs



import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 14  # You can set a base size for the font here
mpl.rcParams['axes.titlepad'] = 20 
mpl.rcParams['axes.labelpad'] = 10

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(lossHist, label='Training Loss', linewidth=2)
plt.plot(valLossHist, label='Validation Loss', linewidth=2)

# Titles and labels with appropriate font sizes
plt.title('Loss Over Epochs', fontsize=18, fontname="Times New Roman")
plt.xlabel('Epoch', fontsize=16, fontname="Times New Roman")
plt.ylabel('Loss', fontsize=16, fontname="Times New Roman")

# Ticks
plt.xticks(fontsize=14, fontname="Times New Roman")
plt.yticks(fontsize=14, fontname="Times New Roman")

plt.legend(loc="upper right", title="Legend", fontsize=14, title_fontsize=16, frameon=True)
plt.grid(True, linestyle='--', alpha=0.7)

# Saving the plot with high resolution
plt.tight_layout()
plt.savefig('Loss_over_epochs_high_res.png', dpi=300)

# Show the plot
plt.show()



##
##
#Losscurve
plt.plot(lossHist, label='Training Loss')
plt.plot(valLossHist, label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


##T-SNE

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# Assuming `sensor_data_values` is a numpy array of shape (8928, 67)
# And `labels_data` is a pandas DataFrame with a timestamp column and the rest being label columns
# Convert the 'Time' column to a datetime format
labels_data['Time'] = pd.to_datetime(labels_data['Time'])

# Convert the datetime to a numeric value (in nanoseconds)
labels_data['timestamp_numeric'] = labels_data['Time'].astype(np.int64)

# Since this will give nanoseconds, we can convert it to seconds by dividing by 1e9 (for visualization purposes)
labels_data['timestamp_numeric'] = labels_data['timestamp_numeric'] // int(1e9)

# Normalize the numeric timestamp to be in the range [0, 1]

scaler = MinMaxScaler()
labels_data['timestamp_numeric'] = scaler.fit_transform(labels_data['timestamp_numeric'].values.reshape(-1, 1))


# Step 2: Reshape the sensor data and concatenate it with the timestamp
reshaped_sensor_data = sensor_data_values.reshape(-1, 1)
timestamps_repeated = np.repeat(labels_data['timestamp_numeric'].values, 67).reshape(-1, 1)
data_with_timestamp = np.hstack([timestamps_repeated, reshaped_sensor_data])

# Step 3: Flatten the labels
flattened_labels = labels_data.iloc[:, 1:-1].values.ravel()  # Exclude the timestamp and timestamp_numeric columns

# Step 4: Apply PCA
data_pca = PCA(n_components=2).fit_transform(data_with_timestamp)

# Assuming all the previous steps have been executed

import matplotlib.pyplot as plt
import matplotlib as mpl

# Set global font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))  # Adjusted height
scatter0 = ax.scatter(data_pca[flattened_labels == 0, 0], data_pca[flattened_labels == 0, 1], 
                      alpha=0.5, edgecolors='w', linewidth=0.5, label="Label 0", color='blue')
scatter1 = ax.scatter(data_pca[flattened_labels == 1, 0], data_pca[flattened_labels == 1, 1], 
                      alpha=0.5, edgecolors='w', linewidth=0.5, label="Label 1", color='red')

# Titles and labels
ax.set_title('PCA of Sensor Data with Timestamp', fontsize=16)
ax.set_xlabel('Principal Component 1', fontsize=14)
ax.set_ylabel('Principal Component 2', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc="upper right", title="Labels", frameon=True)

# Saving with high resolution
plt.tight_layout()
plt.savefig('PCA_plot_high_res.png', dpi=300)

# Show the plot
plt.show()



import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set global font to Times New Roman
mpl.rcParams['font.family'] = 'Times New Roman'

# Apply PCA and T-SNE on data_with_timestamp
data_pca = PCA(n_components=2).fit_transform(data_with_timestamp)
data_tsne = TSNE(n_components=2).fit_transform(data_with_timestamp)

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))  # Adjusted height

# PCA Plot
ax1.scatter(data_pca[flattened_labels == 0, 0], data_pca[flattened_labels == 0, 1], 
            alpha=0.5, edgecolors='w', linewidth=0.5, label="Label 0", color='blue')
ax1.scatter(data_pca[flattened_labels == 1, 0], data_pca[flattened_labels == 1, 1], 
            alpha=0.5, edgecolors='w', linewidth=0.5, label="Label 1", color='red')
ax1.set_title('PCA', fontsize=16)
ax1.set_xlabel('Principal Component 1', fontsize=14)
ax1.set_ylabel('Principal Component 2', fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(loc="upper right", title="Labels", frameon=True)

# T-SNE Plot
ax2.scatter(data_tsne[flattened_labels == 0, 0], data_tsne[flattened_labels == 0, 1], 
            alpha=0.5, edgecolors='w', linewidth=0.5, label="Label 0", color='blue')
ax2.scatter(data_tsne[flattened_labels == 1, 0], data_tsne[flattened_labels == 1, 1], 
            alpha=0.5, edgecolors='w', linewidth=0.5, label="Label 1", color='red')
ax2.set_title('T-SNE', fontsize=16)
ax2.set_xlabel('T-SNE Dimension 1', fontsize=14)
ax2.set_ylabel('T-SNE Dimension 2', fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend(loc="upper right", title="Labels", frameon=True)

# Adjust layout
plt.tight_layout()

# Saving with high resolution
plt.savefig('PCA_and_TSNE_plot_high_res.png', dpi=300)

# Show the plot
plt.show()







# Flatten the data
flattened_sensor_data = sensor_data_values.ravel()
flattened_labels = labels_data.iloc[:, 1:].values.ravel()
reshaped_sensor_data = sensor_data_values.reshape(-1, 67)
# Perform T-SNE and PCA on the data
# Perform T-SNE and PCA on reshaped data
data_tsne = TSNE(n_components=2).fit_transform(reshaped_sensor_data)
data_pca = PCA(n_components=2).fit_transform(reshaped_sensor_data)

# Plot
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=flattened_labels, cmap='viridis', alpha=0.5)
plt.title('T-SNE')

plt.subplot(1, 2, 2)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=flattened_labels, cmap='viridis', alpha=0.5)
plt.title('PCA')

plt.show()











from pyts.datasets import load_basic_motions
from pyts.multivariate.transformation import MultivariateTransformer
from pyts.image import GramianAngularField
X, _, _, _ = load_basic_motions(return_X_y=True)
transformer = MultivariateTransformer(GramianAngularField(),       flatten=False)
X_new = transformer.fit_transform(X)
X_new.shape
(40, 6, 100, 100)



import pandas as pd

# Example multivariate DataFrame
data = {
    'Datetime': pd.date_range('2023-01-01', '2023-01-10'),
    'Temp': [25, 26, 24, 23, 22, 25, 26, 27, 28, 29],
    'Sal': [35, 36, 34, 33, 32, 35, 36, 37, 38, 39],
    'DO_mgl': [8, 7, 9, 8, 7, 8, 9, 10, 11, 10],
    'Turb': [10, 12, 8, 15, 11, 10, 13, 14, 12, 11]
}

multivariate_df = pd.DataFrame(data)
multivariate_df.set_index('Datetime', inplace=True)

# Apply PCA
pca = PCA(n_components=1)  # Set the number of components as needed
latent_variable = pca.fit_transform(multivariate_df)

# Create a DataFrame with the latent variable and datetime index
latent_df = pd.DataFrame(latent_variable, columns=['Latent_Variable'], index=multivariate_df.index)

# Display the resulting DataFrame with the latent variable
print(latent_df)




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D

# Example multivariate DataFrame
data = {
    'Datetime': pd.date_range('2023-01-01', '2023-01-10'),
    'Temp': [25, 26, 24, 23, 22, 25, 26, 27, 28, 29],
    'Sal': [35, 36, 34, 33, 32, 35, 36, 37, 38, 39],
    'DO_mgl': [8, 7, 9, 8, 7, 8, 9, 10, 11, 10],
    'Turb': [10, 12, 8, 15, 11, 10, 13, 14, 12, 11]
}

multivariate_df = pd.DataFrame(data)
multivariate_df.set_index('Datetime', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(multivariate_df)

# Split the data into training and testing sets
X_train, X_test = train_test_split(normalized_data, test_size=0.2, random_state=42)

# Reshape the data for convolutional autoencoder
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Define the convolutional autoencoder model
model = Sequential()
model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(8, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(4, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(2, kernel_size=3, activation='relu', padding='same'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(1, kernel_size=3, activation='relu', padding='same'))

model.add(UpSampling1D(size=2))
model.add(Conv1D(2, kernel_size=3, activation='relu', padding='same'))
model.add(UpSampling1D(size=2))
model.add(Conv1D(4, kernel_size=3, activation='relu', padding='same'))
model.add(UpSampling1D(size=2))
model.add(Conv1D(8, kernel_size=3, activation='relu', padding='same'))
model.add(UpSampling1D(size=2))
model.add(Conv1D(16, kernel_size=3, activation='relu', padding='same'))
model.add(Conv1D(1, kernel_size=3, activation='sigmoid', padding='same'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test))

# Create a new model for encoder part
encoder = Model(inputs=model.input, outputs=model.layers[7].output)

# Encode the data
encoded_data = encoder.predict(normalized_data.reshape((normalized_data.shape[0], normalized_data.shape[1], 1)))

# Extract the latent variable (last layer of the encoder)
latent_variable = encoded_data[:, -1, 0]

# Create a DataFrame with the latent variable and datetime index
latent_df = pd.DataFrame(latent_variable, columns=['Latent_Variable'], index=multivariate_df.index)

# Display the resulting DataFrame with the latent variable
print(latent_df)








#################
##
##
############################################
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split


# Example multivariate DataFrame
data = {
    'Datetime': pd.date_range('2023-01-01', '2023-01-10'),
    'Temp': [25, 26, 24, 23, 22, 25, 26, 27, 28, 29],
    'Sal': [35, 36, 34, 33, 32, 35, 36, 37, 38, 39],
    'DO_mgl': [8, 7, 9, 8, 7, 8, 9, 10, 11, 10],
    'Turb': [10, 12, 8, 15, 11, 10, 13, 14, 12, 11]
}

multivariate_df = pd.DataFrame(data)
multivariate_df.set_index('Datetime', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(multivariate_df)

# Split the data into training and testing sets
X_train, X_test = train_test_split(normalized_data, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train).unsqueeze(1)  # Add a channel dimension
X_test_tensor = torch.Tensor(X_test).unsqueeze(1)  # Add a channel dimension

# Define the autoencoder model
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(4, 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the model, loss function, and optimizer
autoencoder = Autoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

# Training the autoencoder
num_epochs = 500
for epoch in range(num_epochs):
    outputs = autoencoder(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Extract the latent variable
with torch.no_grad():
    latent_variable = autoencoder.encoder(X_train_tensor).squeeze().numpy()

# Create a DataFrame with the latent variable and datetime index
latent_df = pd.DataFrame(latent_variable, columns=['Latent_Variable'], index=multivariate_df.index)

# Display the resulting DataFrame with the latent variable
print(latent_df)
'''





import pandas as pd

# Your original DataFrame
data = {
    'DateTimeStamp': pd.to_datetime(['1/1/2007 0:00', '1/1/2007 0:15', '1/1/2007 0:30', '1/1/2007 0:45',
                                     '1/1/2007 1:00', '1/1/2007 1:15', '1/1/2007 1:30', '1/1/2007 1:45',
                                     '1/1/2007 2:00', '1/1/2007 2:15', '1/1/2007 2:30', '1/1/2007 2:45',
                                     '1/1/2007 3:00', '1/1/2007 3:15', '1/1/2007 3:30', '1/1/2007 3:45']),
    'ACESPWQ_Temp': [17.5, 17.5, 17.4, 17, 16.3, 16.1, 16, 15.9, 15.6, 15.5, 15.5, 15.4, 15.5, 15.4, 15.4, 15.4],
    'ACESPWQ_Sal': [26.8, 26.7, 26.7, 26.4, 26, 25.8, 25.7, 24.8, 24, 23.8, 24.1, 24.7, 26, 27.2, 28.2, 29.3],
    'ACESPWQ_DO_mgl': [4.6, 4.5, 4.6, 4.8, 6.2, 6.3, 6.3, 6.6, 7.2, 7.4, 7.5, 7.6, 7.6, 7.7, 7.7, 7.7],
    'ACESPWQ_Turb': [8, 8, 7, 7, 8, 7, 8, 13, 17, 21, 24, 36, 71, 49, 45, 51]
}

df = pd.DataFrame(data)
df.set_index('DateTimeStamp', inplace=True)

# Resample and calculate the mean for each 2-hour interval
df_resampled = df.resample('2H').mean()

# Display the resulting DataFrame
print(df_resampled)




'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Define the Spatiotemporal GNN model
class SpatiotemporalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpatiotemporalGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Define the layers of the GNN
        self.graph_conv1 = nn.GraphConvolution(input_dim, hidden_dim)
        self.graph_conv2 = nn.GraphConvolution(hidden_dim, hidden_dim)
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, adjacency_matrix):
        x = self.graph_conv1(x, adjacency_matrix)
        x = self.graph_conv2(x, adjacency_matrix)
        x = self.temporal_conv(x)
        x = torch.mean(x, dim=2)
        x = self.fc(x)
        return x

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

# Load the data and labels
data = pd.read_csv('data.csv')
labels = pd.read_csv('label.csv')

# Preprocess the data and labels
# ...

# Create the adjacency matrix
distances = np.array([...])  # Distance matrix of all 19 sensors
adjacency_matrix = 1 / (route_distances+1e-6)
np.fill_diagonal(adjacency_matrix, 0)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)

# Create the custom dataset
train_dataset = CustomDataset(train_data, train_labels)
test_dataset = CustomDataset(test_data, test_labels)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
input_dim = 19  # Number of sensor observations
hidden_dim = 64  # Hidden dimension of the GNN
output_dim = 1  # Number of binary labels
model = SpatiotemporalGNN(input_dim, hidden_dim, output_dim)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data, adjacency_matrix)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
model.eval()
with torch.no_grad():
    for batch_idx, (data, labels) in enumerate(test_loader):
        output = model(data, adjacency_matrix)
        # Calculate evaluation metrics (F1, AUC)
        # ...







import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Step 1: Load Data
data = pd.read_csv("data.csv", parse_dates=["datetime"])
labels = pd.read_csv("label.csv", parse_dates=["datetime"])
distances = route_distances  # Replace ... with your actual distance matrix

# Step 2: Create Graph Representation
# Use the distance matrix to create edges between sensors based on distance criteria.
# Use the reciprocal of distances as edge weights.
adj_matrix = 1 / (distances + 1e-8)  # Adding a small epsilon to avoid division by zero
np.fill_diagonal(adj_matrix, 0)  # Set diagonal to 0

# Convert to PyTorch tensors
adj_matrix = torch.FloatTensor(adj_matrix)
data_values_float = data.values[:, 1:].astype('float64')
data_tensor = torch.FloatTensor(data_values_float)  # Exclude datetime column
label_values_float = labels.values[:, 1:].astype('int32')
labels_tensor = torch.FloatTensor(label_values_float)

# Step 3: Preprocess Data
# Prepare the spatiotemporal features (concatenate data with temporal horizon)
temporal_horizon = 5
num_samples, num_features = data_tensor.shape
X = []
y = []

for i in range(num_samples - temporal_horizon + 1):
    X.append(data_tensor[i:i + temporal_horizon].reshape(-1))
    y.append(labels_tensor[i + temporal_horizon - 1])

X = torch.stack(X)
y = torch.stack(y)


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build Spatiotemporal GNN Model
class SpatioTemporalGraphConvolution(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpatioTemporalGraphConvolution, self).__init__()
        self.gc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.gc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        x = self.gc1(x)
        x = self.relu(x)
        x = torch.mm(adj, x)
        x = self.gc2(x)
        x = self.sigmoid(x)
        return x

input_dim = num_features * temporal_horizon
hidden_dim = 64
output_dim = 1  # Binary classification

model = SpatioTemporalGraphConvolution(input_dim, hidden_dim, output_dim)

# Step 5: Train the Model
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

def train(model, X, y, adj_matrix, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X, adj_matrix)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

import torch
import torch
import torch.sparse

import torch
import torch.sparse

# Assuming num_nodes is the number of nodes in your graph
num_nodes = 19

# Generate a random dense adjacency matrix
adj_matrix_dense = torch.randint(0, 2, size=(num_nodes, num_nodes))

# Create a random sparse adjacency matrix in COO format
nonzero_indices = adj_matrix_dense.nonzero()
values = adj_matrix_dense[nonzero_indices[:, 0], nonzero_indices[:, 1]]

# Create the sparse tensor in COO format
adj_matrix_sparse = torch.sparse.FloatTensor(nonzero_indices.t(), values, torch.Size([num_nodes, num_nodes]))

print(adj_matrix_sparse.to_dense())




train(model, X_train, y_train, adj_matrix_sparse, criterion, optimizer)

# Step 6: Evaluate the Model
model.eval()
with torch.no_grad():
    y_pred = model(X_val, adj_matrix_sparse)

# Convert predictions to binary (0 or 1)
y_pred_binary = (y_pred > 0.5).float()

# Calculate F1 score and AUC
f1 = f1_score(y_val.numpy(), y_pred_binary.numpy())
roc_auc = roc_auc_score(y_val.numpy(), y_pred.numpy())

print(f"F1 Score: {f1}, AUC: {roc_auc}")





for i, col in enumerate(data.columns):
    try:
        np.dtype(data[col])
    except TypeError:
        print(f"Non-uniform data type found in column: {col}")

non_numeric_columns = data.select_dtypes(exclude=['float64']).columns
print(non_numeric_columns)
# Handle missing values by filling them with zeros
data.fillna(0, inplace=True)

# Convert the DataFrame to a PyTorch FloatTensor
data_tensor = torch.FloatTensor(data.values[:, 1:])
import torch

# Convert data values to float64
data_values_float = data.values[:, 1:].astype('float64')

# Convert the NumPy array to a PyTorch FloatTensor
data_tensor = torch.FloatTensor(data_values_float)
