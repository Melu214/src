import pandas as pd
from tensorflow.keras.models import load_model
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F






import tensorflow as tf
tf.random.set_seed(10)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)




from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers



def prepare_estuary_data(mainData, datetime, i):
    """
    Prepare data for a specific estuary.

    Parameters:
    - mainData: DataFrame, the main dataset containing all estuaries data.
    - datetime: Series or list, datetime values to apply to the data.
    - i: int, the estuary index to process data for.

    Returns:
    - A numpy array containing the processed data for the specified estuary.
    """
    # Select the columns for the current estuary based on index i
    X_esturay = mainData.iloc[:, (i-1)*4 : 4*i]
    X_esturay.columns = ['Temp', 'Sal', 'DO', 'Turb']

    # Apply datetime index and sort
    X_esturay['datetime'] = datetime
    X_esturay.index = pd.to_datetime(X_esturay['datetime'], format='%Y-%m-%d %H:%M:%S')
    X_esturay = X_esturay.sort_index()

    # Resample to hourly mean and drop the datetime column
    X_esturay = X_esturay.resample('1H').mean()
    X_esturay = X_esturay.drop(columns=["datetime"])

    # Scale the data
    scaler = MinMaxScaler()
    X_esturay_scaled = scaler.fit_transform(X_esturay)

    # Reshape for LSTM input
    X_esturay_reshaped = X_esturay_scaled.reshape(X_esturay_scaled.shape[0], 1, X_esturay_scaled.shape[1])

    return X_esturay_reshaped, X_esturay_scaled


# Define a single shared encoder for all estuaries
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden.squeeze(0)  # Squeeze to remove the batch dimension

class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.lstm = nn.LSTM(hidden_size, output_size, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm(x)
        return x

# Define the autoencoder model using separate encoders and decoders

class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_estuaries):
        super(Autoencoder, self).__init__()
        self.autoencoders = nn.ModuleList([
            nn.Sequential(
                nn.LSTM(input_size, hidden_size, batch_first=True),
                nn.LSTM(hidden_size, input_size, batch_first=True)
            ) for _ in range(num_estuaries)
        ])

    def forward(self, x, estuary_idx):
        x, _ = self.autoencoders[estuary_idx][0](x)
        x, _ = self.autoencoders[estuary_idx][1](x)
        return x


# Vectorized operation to calculate anomalies
#anomalies = (normalized_errors > 3 * normalized_errors.std()).astype(int)

#labelData[col] = anomalies




def calculate_anomalies(reconstruction_errors):
    """
    Calculate anomalies based on reconstruction errors.

    Parameters:
    - reconstruction_errors: DataFrame, containing reconstruction errors for each feature.

    Returns:
    - A DataFrame with anomalies marked as 1 and normal points as 0.
    """
    anomalies_df = reconstruction_errors.copy()
    for col in anomalies_df.columns:
        # Extract data for the current column
        normalized_errors = anomalies_df[col]
        
        # Vectorized operation to calculate anomalies
        anomalies = (normalized_errors > 3 * normalized_errors.std()).astype(int)
        
        anomalies_df[col] = anomalies
    return anomalies_df


# Function to create sequences from the data
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i: i + sequence_length].numpy()
        sequences.append(sequence)
    return np.array(sequences)


# Assuming you have initialized 'device' earlier
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Example of loading a model and calculating anomalies
folder_path = 'C:/Users/mi0025/Documents/Research/anomalydetector/AutoendoersOutput/'
model_path = folder_path + 'autoencoder_modelEstury5.h5'



reconstruction_errors = pd.read_csv('reconstruction_errors.csv')
reconstruction_errors['datetime'] = pd.to_datetime(reconstruction_errors['datetime'], format='%Y-%m-%d %H:%M:%S')
dateTime = reconstruction_errors['datetime']
reconstruction_errors = reconstruction_errors.drop(columns=["datetime"])
true_labels =  calculate_anomalies(reconstruction_errors)




# Define the date range
start_date = '2007-01-01'
end_date = '2023-09-30'

filename = 'noOutlierData.csv'
#C:/Users/mi0025/Documents/Research/anomalydetector/data_df1.csv

file_path = os.path.join(folder_path, filename)
        
       
        
# Read the CSV file
mainData = pd.read_csv(file_path)
#mainData = mainData[['DateTimeStamp', 'Temp', 'Sal','DO_mgl', 'Turb']]
mainData['datetime'] = pd.to_datetime(mainData['datetime'], format='%m/%d/%Y %H:%M')


mainData.index = mainData['datetime']
datetime = mainData[['datetime']]
mainData = mainData.drop(columns=["datetime"])


# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model




from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

# create the autoencoder model
model = autoencoder_model(X_esturay)
model.compile(optimizer='adam', loss='mae')
model.summary()

#device = tf.device('cuda' if torch.cuda.is_available() else 'cpu')


# fit the model to the data
nb_epochs = 50
batch_size = 512
history = model.fit(X_esturay, X_esturay, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.20).history







# Initialize and train the autoencoder for each estuary
input_size = 4  # You should set this based on your actual input size
hidden_size = 64  # You can adjust this based on your needs
num_estuaries = 19  # Number of estuaries
initial_lr = 0.01  # You can adjust this based on your needs
num_epochs = 50  # You can adjust this based on your needs
batch_size = 512




results = []


i = 1
for estuary in true_labels.columns:
    #read estuary data   
    X_esturay, resampled_data_normalized = prepare_estuary_data(mainData, datetime, i)
    #model_ind = load_model('autoencoder_modelEstury5.h5')
    #X_pred = model.predict(X_esturay)
    
    
    
    # Calculate MSE, RMSE, and MAE
    #mse = mean_squared_error(X_esturay.reshape(X_esturay.shape[0], -1), X_pred.reshape(X_pred.shape[0], -1))
    #rmse = np.sqrt(mse)
    #mae = mean_absolute_error(X_esturay.reshape(X_esturay.shape[0], -1), X_pred.reshape(X_pred.shape[0], -1))
    
    
    data_tensor = torch.tensor(resampled_data_normalized, dtype=torch.float32)
    
    
    # Create sequences
    sequence_length = 10  # You can adjust this based on your needs
    sequences = create_sequences(data_tensor, sequence_length)
    
    # Convert sequences to PyTorch tensor
    sequences = torch.tensor(sequences, dtype=torch.float32)
    
    # Assuming 'device' is defined earlier
    sequences = sequences.to(device)
    
    
    model_com = Autoencoder(input_size, hidden_size, num_estuaries)
    model_com.load_state_dict(torch.load('autoencoder_modelV19Estuaries.pth'))
    model_com.to(device)
    
    with torch.no_grad():
        reconstructed_samples_ind = model_com(sequences, 18)
        # Calculate MSE and MAE for composite model
        mse = F.mse_loss(reconstructed_samples_ind, sequences, reduction='none').mean(dim=2).mean(dim=1).mean().item()
        rmse = np.sqrt(mse)
        mae = F.l1_loss(reconstructed_samples_ind, sequences, reduction='none').mean(dim=2).mean(dim=1).mean().item()
        reconstructed_samples = model_com(sequences, i-1)  # Ensure 'i' is defined or remove it if not necessary
        # Calculate MSE and MAE for composite model
        mse_com = F.mse_loss(reconstructed_samples, sequences, reduction='none').mean(dim=2).mean(dim=1).mean().item()
        rmse_com = np.sqrt(mse_com)
        mae_com = F.l1_loss(reconstructed_samples, sequences, reduction='none').mean(dim=2).mean(dim=1).mean().item()
    
    # Append results
    results.append([estuary, rmse, mae, rmse_com, mae_com])
    i += 1
    
# Create a DataFrame from the results
results_df = pd.DataFrame(results, columns=['Estuary', 'RMSE', 'MAE', 'RMSE_Com', 'MAE_Com'])

# Display or use the DataFrame as needed
print(results_df)

mainData['errors'] = reconstruction_errors.values

#reconstruction_errors = reconstruction_errors.resample('6H').mean()

normalized_errors = (reconstruction_errors - np.min(reconstruction_errors)) / (np.max(reconstruction_errors) - np.min(reconstruction_errors))
anomalies = (normalized_errors > 2 * normalized_errors.std()).astype(int)





#labelData = pd.read_csv('abnormallabels.csv')
labelData['datetime'] = pd.to_datetime(dateTime, format='%Y-%m-%d %H:%M:%S')
labelData.index = labelData['datetime']
labelData = labelData.drop(columns = ['datetime'])








def load_and_evaluate_model(file_path, X_new, Y_new=None):
    """
    Load a saved Keras model and evaluate or predict on a new dataset.

    Parameters:
    - file_path: str, path to the saved model.
    - X_new: numpy array, new input data to evaluate or predict.
    - Y_new: numpy array, optional, new target data for evaluation. If not provided, the function will return predictions.

    Returns:
    - If Y_new is provided, returns the loss and metrics of the model on the new dataset.
    - If Y_new is not provided, returns the model's predictions on X_new.
    """
    # Step 1: Load the saved model
    model = load_model(file_path)

    # Step 2: Evaluate the model on the new dataset or predict if Y_new is not provided
    if Y_new is not None:
        loss, metrics = model.evaluate(X_new, Y_new, verbose=0)
        print(f"Loss: {loss}, Metrics: {metrics}")
        return loss, metrics
    else:
        predictions = model.predict(X_new)
        return predictions

def calculate_anomalies(reconstruction_errors):
    """
    Calculate anomalies based on reconstruction errors.

    Parameters:
    - reconstruction_errors: DataFrame, containing reconstruction errors for each feature.

    Returns:
    - A DataFrame with anomalies marked as 1 and normal points as 0.
    """
    anomalies_df = reconstruction_errors.copy()
    for col in anomalies_df.columns:
        # Extract data for the current column
        normalized_errors = anomalies_df[col]
        
        # Vectorized operation to calculate anomalies
        anomalies = (normalized_errors > 3 * normalized_errors.std()).astype(int)
        
        anomalies_df[col] = anomalies
    return anomalies_df

# Example of loading a model and calculating anomalies
folder_path = 'C:/Users/mi0025/Documents/Research/anomalydetector/AutoendoersOutput/'
model_path = folder_path + 'autoencoder_modelEstury5.h5'

# Assuming X_new is your new dataset for which you want to make predictions
# X_new = ... 

# predictions = load_and_evaluate_model(model_path, X_new)

# Assuming you've already calculated reconstruction errors and they are in 'labelData'
anomalies_df = calculate_anomalies(labelData)
