import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
#matplotlib inline

import h5py
import tensorflow as tf
tf.random.set_seed(10)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from matplotlib import rcParams


from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

from scipy.stats import ttest_ind

###Functions

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

####
def merge_series(arr, threshold):
    # Find unique numbers in the array (excluding 0)
    unique_numbers = list(set(filter(lambda x: x != 0, arr)))

    # Initialize a dictionary to store the start and end indices for each unique number
    indices_dict = {num: {'start': None, 'end': None} for num in unique_numbers}

    # Iterate through the array to find the start and end indices for each unique number
    for i, num in enumerate(arr):
        if num != 0:
            if indices_dict[num]['start'] is None:
                indices_dict[num]['start'] = i
            indices_dict[num]['end'] = i

    # Merge series based on the specified threshold
    for i in range(len(unique_numbers) - 1):
        current_num = unique_numbers[i]
        next_num = unique_numbers[i + 1]

        end_index_current = indices_dict[current_num]['end']
        start_index_next = indices_dict[next_num]['start']
        end_index_next = indices_dict[next_num]['end']

        if start_index_next - end_index_current <= threshold:
            # Merge the series by updating the elements in the specified range
            arr[end_index_current + 1:end_index_next+1] = [current_num] * (end_index_next - end_index_current)

    return arr

# Assuming df is your original DataFrame

def normalize_and_add_count(df, label_column, min_series_length=96, max_gap_between_series=24):
    df['hurricaneCount'] = 0

    # Identify the start of a new series
    start_of_series = (df[label_column] == 1) & (df[label_column].shift(1) == 0)

    # Cumulatively count series
    df['hurricaneCount'] = start_of_series.cumsum()
    df.loc[df[label_column] == 0, 'hurricaneCount'] = 0
    
    df['hurricaneCount'] = merge_series(np.array(df['hurricaneCount']), max_gap_between_series)

    # Calculate the length of each series
    series_lengths = df.groupby('hurricaneCount').size()

    # Identify and drop series with fewer than min_series_length observations
    short_series = series_lengths[series_lengths < min_series_length].index
    # Identify and replace elements of series with fewer than min_series_length observations with 0
    df.loc[df['hurricaneCount'].isin(short_series), 'hurricaneCount'] = 0
    
    
    
    
    # Extract unique values from the column
    unique_values = df['hurricaneCount'].unique()
    
    # Create a mapping of unique values to their positions in the sorted array
    value_to_position = {value: position for position, value in enumerate(sorted(unique_values))}
    
    # Replace the values in the column with their positions
    df['hurricaneCount'] = df['hurricaneCount'].map(value_to_position)

# Now, df[column_name] contains the positions of the unique values


    #df = df[~df['hurricaneCount'].isin(short_series)]
    #input_array = np.array(df['hurricaneCount'])

    
    #df['newLabel'] = result_array

    return df[['hurricaneCount']]  # Return only the newLabel column



def plot_sub_data(sub_data, cid):
    # Features to plot
    features = ['Temp', 'Sal', 'DO_mgl', 'Turb']

    # Time periods
    time_periods = ['pre-cyclone', 'cyclone', 'post-cyclone']

    # Create subplots
    fig, axs = plt.subplots(len(features), 1, figsize=(10, 8), sharex=True)

    # Plot each feature in a subplot
    for i, feature in enumerate(features):
        ax = axs[i]

        # Plot lines for each time period
        for period in time_periods:
            subset = sub_data[sub_data['cyclone_Status'] == period]
            ax.plot(subset['datetime'], subset[feature], label=period)

        # Shade the entire background during the hurricane period
        hurricane_subset = sub_data[sub_data['cyclone_Status'] == 'cyclone']
        hurricane_start = hurricane_subset['datetime'].min()
        hurricane_end = hurricane_subset['datetime'].max()
        ax.axvspan(hurricane_start, hurricane_end, facecolor='red', alpha=0.3, label='Disturbance Window')

        # Add a line for the cumulative mean
        cum_mean = sub_data.groupby('cyclone_Status')[feature].expanding().mean().reset_index(level=0, drop=True)
        ax.plot(sub_data['datetime'], cum_mean, label='Cumulative Mean', linestyle='--', color='black')

        ax.set_ylabel(feature)
        ax.legend()

    plt.tight_layout()

    # Save the plot at 300 DPI
    plt.savefig(f'plot{cid}.png', dpi=300)

    plt.show()


def analyze_sub_data(sub_data):
    # Group by 'Hurricane_Status' and calculate the mean for each feature
    means_by_status = sub_data.groupby('cyclone_Status').mean()

    # Create a table to store results
    results_table = pd.DataFrame(index=['pre-cyclone', 'cyclone', 'post-cyclone'])

    # Loop through each feature and perform t-test
    for feature in ['Temp', 'Sal', 'DO_mgl', 'Turb']:
        pre_hurricane_mean = means_by_status.loc['pre-cyclone', feature]
        hurricane_mean = means_by_status.loc['cyclone', feature]
        post_hurricane_mean = means_by_status.loc['post-cyclone', feature]

        # Perform t-tests
        t_stat_hurricane, p_value_hurricane = ttest_ind(sub_data[sub_data['cyclone_Status'] == 'pre-cyclone'][feature],
                                                         sub_data[sub_data['cyclone_Status'] == 'cyclone'][feature])

        t_stat_post, p_value_post = ttest_ind(sub_data[sub_data['cyclone_Status'] == 'cyclone'][feature],
                                               sub_data[sub_data['cyclone_Status'] == 'post-cyclone'][feature])

        # Store results in the table
        results_table[feature + '_pre_cyclone'] = pre_hurricane_mean
        results_table[feature + '_hcyclone'] = hurricane_mean
        results_table[feature + '_cyclone'] = post_hurricane_mean
        results_table['p_value_cyclone_' + feature] = p_value_hurricane
        results_table['p_value_post_cyclone_' + feature] = p_value_post

    return results_table


#read Main datafile

folder_path = 'C:/Users/mi0025/Documents/Research/anomalydetector/CSVData/'

# Define the date range
start_date = '2007-01-01'
end_date = '2023-09-30'

filename = 'WKBMRWQ.csv'

file_path = os.path.join(folder_path, filename)
        
       
        
# Read the CSV file
mainData = pd.read_csv(file_path, skiprows=2)
mainData = mainData[['DateTimeStamp', 'Temp', 'Sal','DO_mgl', 'Turb']]
mainData['DateTimeStamp'] = pd.to_datetime(mainData['DateTimeStamp'], format='%m/%d/%Y %H:%M')

# Filter the DataFrame to include data within the specified date range
mainData = mainData[(mainData['DateTimeStamp'] >= start_date) & (mainData['DateTimeStamp'] <= end_date)]

mainData.columns = ['datetime', 'Temp', 'Sal', 'DO_mgl', 'Turb']



print(mainData.isna().sum())



# Drop rows with any NaN values
mainData = mainData.dropna()

# If you want to drop rows with blank values as well, you can replace blank values with NaN
mainData.replace("", np.nan, inplace=True)

# Drop rows with any NaN or blank values
mainData = mainData.dropna()

# If you want to reset the index after dropping rows
mainData.reset_index(drop=True, inplace=True)


'''

column_medians = mainData.iloc[:, 1:].median()

column_means = mainData.iloc[:, 1:].mean()

# Replace missing and negative values with column medians
for col in mainData.columns[1:]:
    # Replace missing values with median
    mainData[col] = mainData[col].fillna(column_medians[col])
    
    # Replace negative values with median
    mainData[col] = mainData[col].apply(lambda x: column_medians[col] if x < 0 else x)

print(mainData.isna().sum())
'''



# Impute missing values using linear interpolation


dataset = mainData.copy()

#dataset = pd.read_csv('sampleData.csv')
#dataset_mean_abs = np.array(dataset.abs().mean())
#dataset_mean_abs = pd.DataFrame(dataset_mean_abs.reshape(1,4))
#dataset_mean_abs.index = [filename]
#merged_data = merged_data.append(dataset_mean_abs)
#dataset = dataset.iloc[:,:5]
dataset.index = dataset['datetime']
dataset = dataset.drop(columns=["datetime"])
#merged_data.columns = ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4']

merged_data = dataset.copy()
merged_data.columns = ['Temp', 'Sal', 'DO', 'Turb']

# transform data file index to datetime and sort in chronological order
merged_data.index = pd.to_datetime(merged_data.index, format='%Y-%m-%d %H:%M:%S')
merged_data = merged_data.sort_index()
merged_data.to_csv('Averaged_BearingTest_Dataset.csv')
print("Dataset shape:", merged_data.shape)
merged_data.head()


train = merged_data['2007-01-01 00:00:00': '2018-12-31 23:45:00']
test = merged_data['2019-01-01 00:00:00':]
print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)



fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(train['Turb'], label='Turbidity', color='black', animated = True, linewidth=1)

ax.plot(train['Temp'], label='Temp', color='blue', animated = True, linewidth=1)
ax.plot(train['Sal'], label='Sal', color='red', animated = True, linewidth=1)
ax.plot(train['DO'], label='DO', color='green', animated = True, linewidth=1)
plt.legend(loc='lower left')
ax.set_title('Estuar:WeeksBay Sensor Training Data', fontsize=16)
plt.show()



# transforming data from the time domain to the frequency domain using fast Fourier transform
train_fft = np.fft.fft(train)
test_fft = np.fft.fft(test)




# Assuming 'datetime' is the column representing time in your dataset
# If it's not named 'datetime', replace it with the correct column name

# Subplots for time domain data
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(14, 10), dpi=80)

axes = axes.flatten()

axes[0].plot(train['Turb'], label='Turbidity', color='black', linewidth=1)
axes[0].set_title('Turbidity')

axes[1].plot(train['Temp'], label='Temp', color='blue', linewidth=1)
axes[1].set_title('Temperature')

axes[2].plot(train['Sal'], label='Sal', color='red', linewidth=1)
axes[2].set_title('Salinity')

axes[3].plot(train['DO'], label='DO', color='green', linewidth=1)
axes[3].set_title('Dissolved Oxygen')

for ax in axes:
    ax.legend(loc='lower left')
    ax.set_xlabel('Datetime')
    ax.set_ylabel('Value')

fig.suptitle('Estuar: WeeksBay Sensor Training Data', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# Transforming data from the time domain to the frequency domain using fast Fourier transform
#train_fft = np.fft.fft(train.drop('datetime', axis=1))  # Assuming 'datetime' is the column name
#test_fft = np.fft.fft(test.drop('datetime', axis=1))

# Subplots for frequency domain data
fig_fft, axes_fft = plt.subplots(nrows=4, ncols=1, figsize=(14, 10), dpi=80)

axes_fft = axes_fft.flatten()

axes_fft[0].plot(np.abs(train_fft[:, 0]), label='Turbidity', color='black', linewidth=1)
axes_fft[0].set_title('Turbidity (Frequency Domain)')

axes_fft[1].plot(np.abs(train_fft[:, 1]), label='Temp', color='blue', linewidth=1)
axes_fft[1].set_title('Temperature (Frequency Domain)')

axes_fft[2].plot(np.abs(train_fft[:, 2]), label='Sal', color='red', linewidth=1)
axes_fft[2].set_title('Salinity (Frequency Domain)')

axes_fft[3].plot(np.abs(train_fft[:, 3]), label='DO', color='green', linewidth=1)
axes_fft[3].set_title('Dissolved Oxygen (Frequency Domain)')

for ax_fft in axes_fft:
    ax_fft.legend(loc='upper right')
    ax_fft.set_xlabel('Frequency')
    ax_fft.set_ylabel('Amplitude')

fig_fft.suptitle('Estuar: WeeksBay Sensor Training Data (Frequency Domain)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

###





# normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
scaler_filename = "scaler_data"
joblib.dump(scaler, scaler_filename)

X_merged = scaler.transform(merged_data)

# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)


X_merged = X_merged.reshape(X_merged.shape[0], 1, X_merged.shape[1])

# create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()

#device = tf.device('cuda' if torch.cuda.is_available() else 'cpu')


# fit the model to the data
nb_epochs = 20
batch_size = 10
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history

from tensorflow.keras.models import save_model


# Load the saved model
model.save("autoencoder_modelEstury5.h5")

# Now you can use the loaded_model for predictions or further training

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# ... (your previous code for model definition and training)

# Assuming you have a separate test set X_test
# Calculate predictions on the test set
X_pred = model.predict(X_test)

# Calculate MSE, RMSE, and MAE
mse = mean_squared_error(X_test.reshape(X_test.shape[0], -1), X_pred.reshape(X_pred.shape[0], -1))
rmse = np.sqrt(mse)
mae = mean_absolute_error(X_test.reshape(X_test.shape[0], -1), X_pred.reshape(X_pred.shape[0], -1))

print(f'Mean Squared Error (MSE): {mse}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'Mean Absolute Error (MAE): {mae}')

# Plot the training loss and validation loss over epochs

# Set the font to Times New Roman
rcParams['font.family'] = 'times new roman'
#rcParams['font.family'] = 'times new roman'
rcParams['font.size'] = 16

# Plot the training loss and validation loss over epochs
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss of Estuary Weeksbay')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Set the figure size and DPI
plt.gcf().set_size_inches(5, 3)
plt.savefig('model_loss_plot.png', dpi=300, bbox_inches='tight')  # Save the plot with 600 DPI
plt.show()






X_mPred = model.predict(X_merged)
X_mPred = X_mPred.reshape(X_mPred.shape[0], X_mPred.shape[2])
X_mPred = pd.DataFrame(X_mPred, columns=merged_data.columns)
X_mPred.index = merged_data.index

scoredM = pd.DataFrame(index=merged_data.index)
X_merged = X_merged.reshape(X_merged.shape[0], X_merged.shape[2])


#scoredM['Loss_mae'] = np.mean(np.abs(X_mPred-X_merged), axis = 1)
#scoredM.index = merged_data.index
reconstruction_errors = pd.DataFrame(np.mean(np.abs(X_mPred-X_merged), axis = 1))
reconstruction_errors = reconstruction_errors.resample('2H').mean()

normalized_errors = (reconstruction_errors - np.min(reconstruction_errors)) / (np.max(reconstruction_errors) - np.min(reconstruction_errors))
anomalies = (normalized_errors > 3 * normalized_errors.std()).astype(int)

scoredM = pd.DataFrame(index=anomalies.index)
#scoredM['Threshold'] = 0.09
#scoredM['Anomaly'] = (scoredM['Loss_mae'] > scoredM['Threshold']).astype(int)
scoredM['Estuary_18'] = anomalies

df  = scoredM.copy()

result_df = pd.DataFrame()
label = 'Estuary_18'
# Apply the function for each label
result_df[label] = normalize_and_add_count(df.copy(), label,min_series_length=24, max_gap_between_series=4)['hurricaneCount']

result_df.to_csv('estuary18Labels.csv', header =True, index=True)

result_df = pd.read_csv('estuary18Labels.csv')
result_df['datetime'] = pd.to_datetime(result_df['datetime'], format='%Y-%m-%d %H:%M:%S')
result_df.index = result_df['datetime'] 
result_df = result_df.drop(columns=["datetime"])
df = result_df[['Estuary_18']]











# Step 1: Extract unique values from 'Estuary_18' column excluding 0
unique_values = df['Estuary_18'].unique()
unique_values = unique_values[unique_values != 0]

# Step 2: Create a new dataframe to store the results
result_df1 = pd.DataFrame()

# Step 3: Iterate over unique values and extract data from 'mainData'
# Define an empty results DataFrame
all_results = pd.DataFrame(index=['pre-cyclone', 'cyclone', 'post-cyclone'])

# Step 3: Iterate over unique values and extract data from 'mainData'
for value in unique_values:
    subset = df[df['Estuary_18'] == value]

    if not subset.empty:
        start_time = subset.index[0]
        end_time = subset.index[-1]

        # Extract data from 'mainData' using time windows
        start_window = start_time - pd.Timedelta(days=2)
        end_window = end_time + pd.Timedelta(days=2)

        sub_data = mainData[(mainData['datetime'] >= start_window) & (mainData['datetime'] <= end_window)]

        # Add a new column to indicate the time period
        sub_data['cyclone_Status'] = pd.cut(sub_data['datetime'], bins=[start_window, start_time, end_time, end_window],
                                              labels=['pre-cyclone', 'cyclone', 'post-cyclone'], include_lowest=True)

        #plots
        plot_sub_data(sub_data,value)
        # Perform analysis for the current sub_data
        results_table = analyze_sub_data(sub_data)

        # Append results to the overall results DataFrame
        all_results = pd.concat([all_results, results_table])
        
all_results = all_results.reset_index()
all_results = all_results.drop(columns=["index"])
all_results = all_results.drop_duplicates(keep='first')
all_results = all_results.reset_index()
all_results = all_results.drop(columns=["index"])
all_results = all_results.dropna(how='all')



        
  
        
'''


scoredM = normalize_and_add_count(scoredM)

scoredM.to_csv('sensor1Anom.csv', header=True, index=True)

scoredM.head()

plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.histplot(scoredM['Loss_mae'], bins = 20, kde= True, color = 'blue');
plt.xlim([0.0,.5])




# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index

scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
sns.histplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
plt.xlim([0.0,.5])

# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = 0.275
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()

# calculate the same metrics for the training set 
# and merge all data in a single dataframe for plotting
X_pred_train = model.predict(X_train)
X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
X_pred_train.index = train.index

scored_train = pd.DataFrame(index=train.index)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis = 1)
scored_train['Threshold'] = 0.275
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']
scored = pd.concat([scored_train, scored])


# plot bearing failure time plot
scored.plot(logy=True,  figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red'])

def merge_series(arr, threshold):
    # Find unique numbers in the array (excluding 0)
    unique_numbers = list(set(filter(lambda x: x != 0, arr)))

    # Initialize a dictionary to store the start and end indices for each unique number
    indices_dict = {num: {'start': None, 'end': None} for num in unique_numbers}

    # Iterate through the array to find the start and end indices for each unique number
    for i, num in enumerate(arr):
        if num != 0:
            if indices_dict[num]['start'] is None:
                indices_dict[num]['start'] = i
            indices_dict[num]['end'] = i

    # Merge series based on the specified threshold
    for i in range(len(unique_numbers) - 1):
        current_num = unique_numbers[i]
        next_num = unique_numbers[i + 1]

        end_index_current = indices_dict[current_num]['end']
        start_index_next = indices_dict[next_num]['start']
        end_index_next = indices_dict[next_num]['end']

        if start_index_next - end_index_current <= threshold:
            # Merge the series by updating the elements in the specified range
            arr[end_index_current + 1:end_index_next+1] = [current_num] * (end_index_next - end_index_current)

    return arr

# Example usage:
input_array = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0,0,0, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0,4,4,4,0,0,0]
threshold_value = 5
result_array = merge_series(input_array, threshold_value)

print(result_array)

'''
