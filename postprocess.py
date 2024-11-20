# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
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

def normalize_and_add_count(df, label_column, min_series_length=24, max_gap_between_series=24):
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





data = pd.read_csv("reconstruction_errorsv2.csv", parse_dates=["datetime"])
label_data = pd.DataFrame()

for estuary_idx in data.columns[1:]:
    print(estuary_idx)
    
    normalized_errors = data[estuary_idx].values
    
    anomalies = (normalized_errors > 3 * normalized_errors.std()).astype(int)
    
    label_data[estuary_idx] = anomalies
    
label_data['datetime'] = data['datetime']


# Convert datetime column to datetime format
label_data['datetime'] = pd.to_datetime(label_data['datetime'])

# Extract datetime column
datetime_column = label_data["datetime"]
# Drop datetime column for processing
label_data = label_data.drop(columns=["datetime"])
# List of labels from label1 to label19
labels = label_data.columns

# Create a new DataFrame for the results
result_df = pd.DataFrame()

# Apply the function for each label
for label in labels:
    result_df[label] = normalize_and_add_count(label_data.copy(), label)['hurricaneCount']
    
result_df.index = datetime_column[:result_df.shape[0]]
result_df.to_csv("temp1.csv",header=True, index=True)

df = result_df[['Estuary_18']]
#WeeksBay Estuary


#read Main datafile

folder_path = 'C:/Users/mi0025/Documents/Research/anomalydetector/CSVData/'

# Define the date range
start_date = '2007-01-01'
end_date = '2023-09-30'

filename = 'WKBMRWQ.csv'
import os
file_path = os.path.join(folder_path, filename)
        
       
        
# Read the CSV file
mainData = pd.read_csv(file_path, skiprows=2)
mainData = mainData[['DateTimeStamp', 'Temp', 'Sal','SpCond','DO_pct','pH', 'DO_mgl', 'Turb']]
mainData['DateTimeStamp'] = pd.to_datetime(mainData['DateTimeStamp'], format='%m/%d/%Y %H:%M')

# Filter the DataFrame to include data within the specified date range
mainData = mainData[(mainData['DateTimeStamp'] >= start_date) & (mainData['DateTimeStamp'] <= end_date)]

mainData.columns = ['datetime', 'Temp', 'Sal','SpCond','DO_pct','pH', 'DO_mgl', 'Turb']


from fancyimpute import IterativeImputer

for col in mainData.columns[1:]:
    mainData[col] = mainData[col].interpolate()
    imputer = IterativeImputer()
    mainData[col] = imputer.fit_transform(mainData[col].values.reshape(-1, 1))


# Step 1: Extract unique values from 'Estuary_18' column excluding 0
unique_values = df['Estuary_18'].unique()
unique_values = unique_values[unique_values != 0]

# Step 2: Create a new dataframe to store the results
result_df1 = pd.DataFrame()

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
        sub_data['Hurricane_Status'] = pd.cut(sub_data['datetime'], bins=[start_window, start_time, end_time, end_window],
                                              labels=['pre-hurricane', 'hurricane', 'post-hurricane'], include_lowest=True)

        result_df1 = pd.concat([result_df1, sub_data])

# Step 4: Reset the index
result_df1 = result_df1.reset_index()

# Step 5: Display the resulting dataframe
print(result_df1)

