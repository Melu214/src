# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 13:47:56 2023

@author: Usama
"""
'''
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv

# Generate random data for testing
num_sensors = 19
num_variables = 4
num_samples = 100  # Adjust this based on your actual data size

# Generate random features_data tensor
features_data = np.random.rand(num_sensors, num_variables, num_samples)
features_tensor = torch.tensor(features_data, dtype=torch.float32)

# Generate random labels_data tensor
labels_data = np.random.rand(num_sensors, num_samples)
labels_tensor = torch.tensor(labels_data, dtype=torch.float32)

# Generate random adjacency_matrix
adjacency_matrix = np.random.rand(num_sensors, num_sensors)
adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

adjacency_matrix = torch.tensor((adjacency_matrix != 0), dtype=torch.long)
edge_indices = torch.nonzero(adjacency_matrix, as_tuple=False).t()


# Convert data to PyTorch tensors
features_tensor = torch.tensor(features_data, dtype=torch.float32)
labels_tensor = torch.tensor(labels_data, dtype=torch.float32)
adjacency_matrix = torch.tensor(adjacency_matrix, dtype=torch.float32)

# Create edge indices from the adjacency matrix
edge_indices = torch.nonzero(adjacency_matrix, as_tuple=False).t()

# Define a GNN model
class GraphNeuralNetwork(nn.Module):
    def __init__(self, num_features, num_nodes):
        super(GraphNeuralNetwork, self).__init__()
        self.conv1 = GraphConv(num_features, 64)
        self.conv2 = GraphConv(64, 32)
        self.fc1 = nn.Linear(32 * num_nodes, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x

# Instantiate the model
num_features = 4  # Number of variables for each sensor
num_sensors = 19
model = GraphNeuralNetwork(num_features, num_sensors)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    output = model(features_tensor, edge_indices)

    # Compute the loss
    loss = criterion(output, labels_tensor.view(-1))

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
        
        
        
zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)



'''


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data




# Set a random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Number of nodes and features
num_nodes = 4
num_features = 3
sample_length = 200

# Generate random feature matrix A
A = np.random.rand(num_nodes, num_features, sample_length)

# Generate random edge index B (assuming a fully connected graph for simplicity)
edge_index = np.array([[i, j] for i in range(num_nodes) for j in range(num_nodes) if i != j]).T

# Generate random label matrix y
y = np.random.randint(2, size=(num_nodes, sample_length))

# Convert data to PyTorch tensors
A_tensor = torch.tensor(A, dtype=torch.float32)
B_tensor = torch.tensor(edge_index, dtype=torch.long)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create a PyTorch Geometric Data object
data = Data(x=A_tensor, edge_index=B_tensor, y=y_tensor)

# Print the generated data shapes
print("Feature Matrix (A) Shape:", A.shape)
print("Edge Index (B) Shape:", edge_index.shape)
print("Label Matrix (y) Shape:", y.shape)


# Create a PyTorch Geometric Data object
#data = Data(x=A_tensor, edge_index=B_tensor, y=y_tensor)

# Define a simple Graph Neural Network (GNN) model
class GNNModel(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.sigmoid(x)

# Instantiate the model
num_features = A.shape[1]  # Number of features per node
hidden_dim = 64  # Hidden dimension in the GNN
num_classes = 1  # Output dimension (assuming binary classification)

model = GNNModel(num_features, hidden_dim, num_classes)

# Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output.view(-1), y_tensor.view(-1))
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# After training, you can use the trained model to make predictions on new data
model.eval()
with torch.no_grad():
    predictions = model(data).cpu().numpy()

# 'predictions' now contains the predicted labels for your nodes





import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader

# Assume you have node_features, edge_index, and labels for your graph
# node_features: (num_nodes, num_features)
# edge_index: (2, num_edges) - assuming an undirected graph
# labels: (num_nodes, )

# Sample data creation
node_features = torch.randn(4, 3, 200)  # Assuming 4 nodes, each with 3 features of length 200
edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
labels = torch.randint(2, (4, ), dtype=torch.float)  # Random binary labels

# Create a PyTorch Geometric Data object
data = Data(x=node_features, edge_index=edge_index, y=labels)

# Define a simple Graph Neural Network model
class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(node_features.size(1), 16)  # Assuming 16 hidden units
        self.conv2 = GCNConv(16, 2)  # 2 classes for binary classification

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Instantiate the model, loss function, and optimizer
model = GNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data.y.long())  # Assuming binary classification
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Make predictions on the test set
model.eval()
with torch.no_grad():
    pred = model(data)

# Convert predictions to class labels
predicted_labels = pred.argmax(dim=1)
print("Predicted Labels:", predicted_labels)


node_feats = torch.arange(8, dtype=torch.float32).view(1, 4, 2)
adj_matrix = torch.Tensor([[[1, 1, 0, 0],
                            [1, 1, 1, 1],
                            [0, 1, 1, 1],
                            [0, 1, 1, 1]]])

print("Node features:\n", node_feats)
print("\nAdjacency matrix:\n", adj_matrix)




class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True)
        node_feats = self.projection(node_feats)
        node_feats = torch.bmm(adj_matrix, node_feats)
        node_feats = node_feats / num_neighbours
        return node_feats
    
    
    
    
layer = GCNLayer(c_in=2, c_out=2)
layer.projection.weight.data = torch.Tensor([[1., 0.], [0., 1.]])
layer.projection.bias.data = torch.Tensor([0., 0.])

with torch.no_grad():
    out_feats = layer(node_feats, adj_matrix)

print("Adjacency matrix", adj_matrix)
print("Input features", node_feats)
print("Output features", out_feats)








# names of hurricanes
names = ['Cuba I', 'San Felipe II Okeechobee', 'Bahamas', 'Cuba II', 'CubaBrownsville', 'Tampico', 'Labor Day', 'New England', 'Carol', 'Janet', 'Carla', 'Hattie', 'Beulah', 'Camille', 'Edith', 'Anita', 'David', 'Allen', 'Gilbert', 'Hugo', 'Andrew', 'Mitch', 'Isabel', 'Ivan', 'Emily', 'Katrina', 'Rita', 'Wilma', 'Dean', 'Felix', 'Matthew', 'Irma', 'Maria', 'Michael']
 
# months of hurricanes
months = ['October', 'September', 'September', 'November', 'August', 'September', 'September', 'September', 'September', 'September', 'September', 'October', 'September', 'August', 'September', 'September', 'August', 'August', 'September', 'September', 'August', 'October', 'September', 'September', 'July', 'August', 'September', 'October', 'August', 'September', 'October', 'September', 'September', 'October']
 
# years of hurricanes
years = [1924, 1928, 1932, 1932, 1933, 1933, 1935, 1938, 1953, 1955, 1961, 1961, 1967, 1969, 1971, 1977, 1979, 1980, 1988, 1989, 1992, 1998, 2003, 2004, 2005, 2005, 2005, 2005, 2007, 2007, 2016, 2017, 2017, 2018]
 
# maximum sustained winds (mph) of hurricanes
max_sustained_winds = [165, 160, 160, 175, 160, 160, 185, 160, 160, 175, 175, 160, 160, 175, 160, 175, 175, 190, 185, 160, 175, 180, 165, 165, 160, 175, 180, 185, 175, 175, 165, 180, 175, 160]
 
# areas affected by each hurricane
areas_affected = [['Central America', 'Mexico', 'Cuba', 'Florida', 'The Bahamas'], ['Lesser Antilles', 'The Bahamas', 'United States East Coast', 'Atlantic Canada'], ['The Bahamas', 'Northeastern United States'], ['Lesser Antilles', 'Jamaica', 'Cayman Islands', 'Cuba', 'The Bahamas', 'Bermuda'], ['The Bahamas', 'Cuba', 'Florida', 'Texas', 'Tamaulipas'], ['Jamaica', 'Yucatn Peninsula'], ['The Bahamas', 'Florida', 'Georgia', 'The Carolinas', 'Virginia'], ['Southeastern United States', 'Northeastern United States', 'Southwestern Quebec'], ['Bermuda', 'New England', 'Atlantic Canada'], ['Lesser Antilles', 'Central America'], ['Texas', 'Louisiana', 'Midwestern United States'], ['Central America'], ['The Caribbean', 'Mexico', 'Texas'], ['Cuba', 'United States Gulf Coast'], ['The Caribbean', 'Central America', 'Mexico', 'United States Gulf Coast'], ['Mexico'], ['The Caribbean', 'United States East coast'], ['The Caribbean', 'Yucatn Peninsula', 'Mexico', 'South Texas'], ['Jamaica', 'Venezuela', 'Central America', 'Hispaniola', 'Mexico'], ['The Caribbean', 'United States East Coast'], ['The Bahamas', 'Florida', 'United States Gulf Coast'], ['Central America', 'Yucatn Peninsula', 'South Florida'], ['Greater Antilles', 'Bahamas', 'Eastern United States', 'Ontario'], ['The Caribbean', 'Venezuela', 'United States Gulf Coast'], ['Windward Islands', 'Jamaica', 'Mexico', 'Texas'], ['Bahamas', 'United States Gulf Coast'], ['Cuba', 'United States Gulf Coast'], ['Greater Antilles', 'Central America', 'Florida'], ['The Caribbean', 'Central America'], ['Nicaragua', 'Honduras'], ['Antilles', 'Venezuela', 'Colombia', 'United States East Coast', 'Atlantic Canada'], ['Cape Verde', 'The Caribbean', 'British Virgin Islands', 'U.S. Virgin Islands', 'Cuba', 'Florida'], ['Lesser Antilles', 'Virgin Islands', 'Puerto Rico', 'Dominican Republic', 'Turks and Caicos Islands'], ['Central America', 'United States Gulf Coast (especially Florida Panhandle)']]
 
# damages (USD($)) of hurricanes
damages = ['Damages not recorded', '100M', 'Damages not recorded', '40M', '27.9M', '5M', 'Damages not recorded', '306M', '2M', '65.8M', '326M', '60.3M', '208M', '1.42B', '25.4M', 'Damages not recorded', '1.54B', '1.24B', '7.1B', '10B', '26.5B', '6.2B', '5.37B', '23.3B', '1.01B', '125B', '12B', '29.4B', '1.76B', '720M', '15.1B', '64.8B', '91.6B', '25.1B']
 
# deaths for each hurricane
deaths = [90,4000,16,3103,179,184,408,682,5,1023,43,319,688,259,37,11,2068,269,318,107,65,19325,51,124,17,1836,125,87,45,133,603,138,3057,74]
 
# write your update damages function here:
#Convert numerical entries to floats
def update_damages(damages):
  updated_damages = []
  for damage in damages:
    if damage == 'Damages not recorded':
      updated_damages.append(damage)
    elif damage[-1] == 'B':
      updated_damages.append(float(damage.strip('B'))*1000000000)
    else:
      updated_damages.append(float(damage.strip('M'))*1000000)
  return updated_damages
 
updated_damages = update_damages(damages)
 
# write your construct hurricane dictionary function here:
def construct_hurricane(names, months, years, maxwind, areas, damage, death):
  values = []
  for i in range(len(names)):
    #create a new dictionary for each hurricane and add corresponding key:values from each list
    hurricane = {}
    hurricane.update({'Name':names[i], 'Month':months[i], 'Year':years[i], 'Max Sustained Wind':maxwind[i], 'Areas Affected':areas[i], 'Damage':updated_damages[i], 'Deaths':deaths[i]})
    values.append(hurricane)
  #create finalised dictionary - key = name, value = dictionary of hurricane info
  hurr_data = {key:value for key,value in zip(names, values)}
  return hurr_data
 
hurricane_dictionary = construct_hurricane(names, months, years, max_sustained_winds, areas_affected, updated_damages, deaths)
 
# write your construct hurricane by year dictionary function here:
 
def hurricane_by_year(hurricanes):
  hurricane_years = {}
  unique_years = []
  #identify unique years as keys, then add dictionaries of hurricanes that occurred that year as values
  for key in hurricanes:
    working = [hurricanes.get(key)]
    year = hurricanes[key].get('Year')
    if year not in unique_years:
      unique_years.append(year)
      hurricane_years.update({year:working})
    else:
      hurricane_years[year].append(hurricanes.get(key))
  return hurricane_years
 
hurricanes_by_year = hurricane_by_year(hurricane_dictionary)
#print(hurricanes_by_year[1932])
 
# write your count affected areas function here:
def area_count(dictionary):
  unique = []
  counts = []
  #maintain a list of areas as each is encountered, and a corresponding list of counts that can be zipped and returned as a dictionary
  for key in dictionary:
    areas = dictionary[key].get('Areas Affected')
    for area in areas:
      if area not in unique:
        unique.append(area)
        counts.append(1)
      else:
        x = unique.index(area)
        counts[x] += 1
  area_counts = {key:value for key, value in zip(unique,counts)}
  return area_counts
 
area_count = area_count(hurricane_dictionary)
 
# write your find most affected area function here:
def most_affected(dictionary):
  highest = 0
  area = ''
  #maintain variables containing the highest count and area encountered so far as we iterate through
  for key in dictionary:
    if dictionary[key] > highest:
      highest = dictionary[key]
      area = key
  print('{loc} is the most affected area, with {count} hurricanes.'.format(loc=area, count=highest))
 
#most_affected(area_count)
 
# write your greatest number of deaths function here:
 
def most_deadly(dictionary):
  highest = 0
  name = ''
  #logic virtually identical to the most affected function
  for key in dictionary:
    if dictionary[key].get('Deaths') > highest:
      highest = dictionary[key].get('Deaths')
      name = key
  print('The most deadly hurricane was {hurricane}, causing {deaths} deaths.'.format(hurricane=name,deaths=highest))
 
#most_deadly(hurricane_dictionary)
 
# write your catgeorize by mortality function here:
def hurricane_mortality_rating(dictionary):
  mort_rates = {0:[], 1:[], 2:[], 3:[], 4:[]}
  mort_scale = {0: 0, 1: 100, 2: 500, 3:1000, 4:10000}
  for hurr in dictionary:
    death = dictionary[hurr].get('Deaths')
    for x in range(len(mort_scale)):
      #check if deaths are in the currently iterated band, if so update the rate dictionary and break
      if death <= mort_scale[x]:
        mort_rates[x].append(dictionary[hurr])
        break
      else:
        continue
  return mort_rates
 
hurricane_by_mortality_rating = hurricane_mortality_rating(hurricane_dictionary)
 
# write your greatest damage function here:
def greatest_damage(dictionary):
  highest = 0
  name = ''
  #logic similar to most deadly but need to catch the strings!
  for key in dictionary:
    if dictionary[key].get('Damage') == 'Damages not recorded':
      continue
    elif dictionary[key].get('Damage') > highest:
      highest = dictionary[key].get('Damage')
      name = key
  print('The most damaging hurricane was {name}, costing ${cost}.'.format(name=name, cost=highest))
 
#greatest_damage(hurricane_dictionary)
 
# write your catgeorize by damage function here:
def hurricane_damage_rating(dictionary):
  dam_rates = {0:[], 1:[], 2:[], 3:[], 4:[], 5:[], 'Not Recorded':[]}
  dam_scale = {0: 0, 1: 100000000, 2: 1000000000, 3:10000000000, 4:50000000000, 5:1000000000000}
  #same as mortality rating function, but extra line to catch the strings 
  for hurr in dictionary:
    damage = dictionary[hurr].get('Damage')
    if type(damage)== str:
      dam_rates['Not Recorded'].append(dictionary[hurr])
      continue
    for x in range(len(dam_scale)):
      if damage <= dam_scale[x]:
        dam_rates[x].append(dictionary[hurr])
        break
      else:
        continue
  return dam_rates
 
hurricane_by_damage_rating = hurricane_damage_rating(hurricane_dictionary)