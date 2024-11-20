import pandas as pd

def readData(filename):
    #filename: Name of CSV file
    df = pd.read_csv(filename,skiprows=2)
    #choose the relevent features
    df  = df[[['DateTimeStamp','Temp',  'Sal', 'DO_mgl', 'Turb']]]
    
    
    

dir = 'C:/Users/Usama/Documents/Research/anomalydetector/CSVData/RKBLHWQ.csv'
df = pd.read_csv(dir, skiprows=2)
df = df[['DateTimeStamp','Temp',  'Sal', 'DO_mgl', 'Turb']]
#df = df[['DateTimeStamp','Sal']]

df['DateTimeStamp'] = pd.to_datetime(df['DateTimeStamp'], format='%m/%d/%Y %H:%M')



#df.dropna(how='any', inplace=True)


# Define the date range
start_date = '2007-01-01'
end_date = '2023-09-30'

# Filter the DataFrame to include data within the specified date range
df = df[(df['DateTimeStamp'] >= start_date) & (df['DateTimeStamp'] <= end_date)]

df.describe()






from fancyimpute import IterativeImputer
cols = df.columns
for i in cols[1:]:
    df[i] = df[i].interpolate()
    imputer = IterativeImputer()
    df[i] = imputer.fit_transform(df[i].values.reshape(-1, 1))

# Assuming df is your DataFrame
missing_values_per_column = df.isna().sum()

# Print the missing values count for each column
print(missing_values_per_column)    
    
df.set_index('DateTimeStamp', inplace=True)


import pandas as pd

# Assuming you have a DataFrame 'df' with columns: 'DateTimeStamp' and 'Turbidity'
# You can load your data into 'df' here

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
    turbidity = row['Turb']

    # Check if a potential hurricane event is starting
    if turbidity > rise_threshold and hurricane_start is None:
        hurricane_start = index

    # Check if a potential hurricane event is ending
    elif (turbidity < rise_threshold and hurricane_start is not None ):
        if((index - hurricane_start).seconds / 3600 < min_duration):
            hurricane_start = None
        else:
            hurricane_end = index
            actual_start = hurricane_start - pd.Timedelta(hours=20)
            actual_end = hurricane_end + pd.Timedelta(hours=20)
            # Label the data points within the hurricane event as 1
            df.loc[actual_start:actual_end, 'Hurricane_Label'] = 1
    
            # Reset the hurricane_start variable
            hurricane_start = None








from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Assuming your data is stored in a DataFrame called 'sensor_data'
# Drop any unnecessary columns
selected_features = df[['Temp', 'Sal', 'DO_mgl', 'Turb']]

# Handle missing values if needed
# selected_features = selected_features.dropna()

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(selected_features)

# Apply PCA to reduce dimensionality
pca = PCA(n_components=4)  # Adjust the number of components as needed
pca_result = pca.fit_transform(normalized_data)

# Choose the number of clusters (K) based on the PCA components
k = 2

# Apply K-means clustering on the PCA components
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(pca_result)

# Identify anomalies based on distance to cluster centroid
df['Anomaly'] = kmeans.transform(pca_result).mean(axis=1)


# Drop intermediate columns if not needed
#df.drop(['Cluster',       'Anomaly'], axis=1, inplace=True)






import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'Cluster' column
plt.figure(figsize=(8, 6))

# Plot the clusters based on the first two principal components
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Hurricane_Label'], cmap='viridis', edgecolors='k', alpha=0.7)

# Add labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Clustering')

# Show the plot
plt.show()





from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Assuming your data is stored in a DataFrame called 'sensor_data'
# Drop any unnecessary columns
selected_features = df[['Temp', 'Sal', 'DO_mgl', 'Turb']]

# Handle missing values if needed
# selected_features = selected_features.dropna()

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(selected_features)

# Apply t-SNE to reduce dimensionality
tsne = TSNE(n_components=2, random_state=42)  # Adjust the number of components as needed
tsne_result = tsne.fit_transform(normalized_data)

# Choose the number of clusters (K) based on t-SNE components
k = 2

# Apply K-means clustering on the t-SNE components
kmeans = KMeans(n_clusters=k, random_state=42)
df['Cluster'] = kmeans.fit_predict(tsne_result)

# Identify anomalies based on distance to cluster centroid
df['Anomaly'] = kmeans.transform(tsne_result).mean(axis=1)

# Plot the t-SNE result with cluster colors
plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 1], tsne_result[:, 2], c=df['Hurricane_Label'], cmap='viridis', edgecolors='k', alpha=0.7)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Clustering')
plt.show()





















from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Assuming your data is stored in a DataFrame called 'sensor_data'
# Drop any unnecessary columns
selected_features = df[['Temp', 'Sal', 'DO_mgl', 'Turb']]

# Assign more weight to 'Turb' by multiplying its values
selected_features['Turb'] = selected_features['Turb'] **2  # Replace 'some_factor' with the desired weight

# Handle missing values if needed
# selected_features = selected_features.dropna()

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(selected_features)

# Choose the number of clusters (K)
k = 2

# Define a weighted distance function using the Mahalanobis distance
def weighted_distance(X, centers, weights):
    diff = X[:, np.newaxis] - centers
    distance = np.einsum('ijk,ijk->ij', diff, diff)  # Mahalanobis distance
    return np.sqrt(np.sum(distance * weights, axis=1))  # Adjust the shape of weights

# Compute the weighted distances
weights = np.ones((normalized_data.shape[0], k))
weights[:, -1] = 10  # Applying the weight to the 'Turb' column

# Apply K-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(normalized_data)

# Manually compute cluster centers
cluster_centers = kmeans.cluster_centers_

# Assign cluster labels to the original DataFrame
df['Cluster'] = kmeans.labels_

# Identify anomalies based on distance to cluster centroid
df['Anomaly'] = weighted_distance(normalized_data, cluster_centers, weights)







import pickle

# Assuming your Pickle file is named 'your_data.pkl'
file_path = 'C:/Users/Usama/Documents/Research/anomalydetector/data/omi-1_test_label.pkl'

# Read Pickle data
# Read Pickle data with multiple objects
with open(file_path, 'rb') as file:
    obj1	=	pickle.load(file)
    obj2	=	pickle.load(file)
    obj3	=	pickle.load(file)
    obj4	=	pickle.load(file)
    obj5	=	pickle.load(file)
    obj6	=	pickle.load(file)
    obj7	=	pickle.load(file)
    obj8	=	pickle.load(file)
    obj9	=	pickle.load(file)
    obj10	=	pickle.load(file)
    obj11	=	pickle.load(file)
    obj12	=	pickle.load(file)
    obj13	=	pickle.load(file)
    obj14	=	pickle.load(file)
    obj15	=	pickle.load(file)
    obj16	=	pickle.load(file)
    obj17	=	pickle.load(file)
    obj18	=	pickle.load(file)
    obj19	=	pickle.load(file)
    obj20	=	pickle.load(file)
    obj21	=	pickle.load(file)


# Now, 'your_data' contains the data stored in the Pickle file

for cur_run in range(1):
    print('************************ run ************************')
    print(cur_run)
    print('************************     ************************')

    # Load Data
    fname = 'data/' + str(cur_run) + '.pickle'
    with open(fname, 'rb') as f:
        data_dic = pickle.load(f)
    x_train, g_train, y_train = data_dic['x_train'], data_dic['g_train'], data_dic['y_train']
    x_val, g_val, y_val = data_dic['x_val'], data_dic['g_val'], data_dic['y_val']
    x_test, g_test, y_test = data_dic['x_test'], data_dic['g_test'], data_dic['y_test']

    # Convert data to tensor
    x_train, g_train, y_train, x_val, g_val, y_val, x_test, g_test, y_test = totensor(x_train), totensor(g_train), totensor(y_train), totensor(x_val), totensor(g_val), totensor(y_val), totensor(x_test), totensor(g_test), totensor(y_test)

   