# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 22:28:31 2023

@author: Usama
"""
'''
import pandas as pd

dir = 'C:/Users/Usama/Documents/Research/anomalydetector/CSVData/RKBLHWQ.csv'
df = pd.read_csv(dir, skiprows=2)
df = df[['DateTimeStamp','Temp',  'Sal', 'DO_mgl']]
#df = df[['DateTimeStamp','Sal']]

df['DateTimeStamp'] = pd.to_datetime(df['DateTimeStamp'], format='%m/%d/%Y %H:%M')



#df.dropna(how='any', inplace=True)


# Define the date range
start_date = '2022-01-01'
end_date = '2022-12-31'

# Filter the DataFrame to include data within the specified date range
df = df[(df['DateTimeStamp'] >= start_date) & (df['DateTimeStamp'] <= end_date)]

df.describe()


# Assuming df is your DataFrame
missing_values_per_column = df.isna().sum()

# Print the missing values count for each column
print(missing_values_per_column)



from fancyimpute import IterativeImputer

for i in df.columns:
    df[i] = df[i].interpolate()
    imputer = IterativeImputer()
    df[i] = imputer.fit_transform(df[i].values.reshape(-1, 1))



from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Assuming you have a DataFrame 'df' with columns: 'DateTimeStamp', 'Temp', 'Sal', 'DO_mgl'
# You can load your data into 'df' here

# Select 'DateTimeStamp' and 'Sal' columns
sal_df = df[['DateTimeStamp', 'Sal']]

# Set 'DateTimeStamp' as the index
sal_df.set_index('DateTimeStamp', inplace=True)

# Apply exponential smoothing
model = ExponentialSmoothing(sal_df['Sal'], seasonal='add', seasonal_periods=24*4)
fit = model.fit(optimized=True, use_brute=True)

# Forecast or smooth the time series
smoothed_sal = fit.fittedvalues

# Add the smoothed Sal column back to the DataFrame
df['Smoothed_Sal'] = smoothed_sal






import plotly.express as px
import pandas as pd

# Assuming you have a DataFrame 'df' with columns: 'DateTimeStamp', 'Temp', 'Sal', 'DO_mgl'
# You can load your data into 'df' here

# Create an interactive plot using Plotly

# Assuming you have a DataFrame 'df' with columns: 'DateTimeStamp', 'Temp', 'Sal', 'DO_mgl'
# You can load your data into 'df' here

# Create an interactive plot using Plotly
fig = px.line(df, x='DateTimeStamp', y=['Temp', 'Sal', 'DO_mgl'], title='Temperature, Salinity, and DO over Time')
fig.update_xaxes(
    rangeselector=dict(
        buttons=list([
            dict(count=1, label="1d", step="day", stepmode="backward"),
            dict(count=7, label="1w", step="day", stepmode="backward"),
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    ),
    rangeslider=dict(visible=True)
)

fig.update_layout(xaxis_title='DateTimeStamp', yaxis_title='Value')

fig.show()







#new_column_names = {'DateTimeStamp': 'timestamp', 'Sal': 'value'}

# Use the rename method to change column names
#df = df.rename(columns=new_column_names)

df.index = df['DateTimeStamp']

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named df with an index 'DateTimeStamp'
x = df['Sal']  # The x-values will be the index
y = smoothed_sal  # Replace 'your_column_name' with the actual column you want to plot

sns.lineplot(x=x, y=y)
plt.show()



import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.express as px

# Assuming 'filtered_df' is your DataFrame after filtering the date range
# Convert 'DateTimeStamp' to a datetime data type
#filtered_df['DateTimeStamp'] = pd.to_datetime(filtered_df['DateTimeStamp'], format='%m/%d/%Y %H:%M')
#filtered_df.set_index('DateTimeStamp', inplace=True)

# Define the columns you want to decompose
columns_to_decompose = ['Temp', 'Sal', 'DO_mgl']

# Initialize a Plotly figure
fig = None

# Decompose and plot each column
for column in columns_to_decompose:
    decomposition = seasonal_decompose(df[column], model='additive', period=4*24)  # Assuming yearly seasonality
    decomposed_df = pd.DataFrame({
        'Observed': decomposition.observed,
        'Trend': decomposition.trend,
        'Seasonal': decomposition.seasonal,
        'Residual': decomposition.resid
    })

    # Create a Plotly figure
    if fig is None:
        fig = px.line(decomposed_df, x=decomposed_df.index, y=decomposed_df.columns, title=f'Decomposition for {column}')
        fig.update_xaxes(title_text='Date')
        fig.update_yaxes(title_text='Value')
    else:
        new_fig = px.line(decomposed_df, x=decomposed_df.index, y=decomposed_df.columns, title=f'Decomposition for {column}')
        new_fig.update_xaxes(title_text='Date')
        new_fig.update_yaxes(title_text='Value')

        # Add new traces to the existing figure
        for trace in new_fig.data:
            fig.add_trace(trace)

# Show the interactive plot
fig.show()





from msanomalydetector import SpectralResidual
from msanomalydetector import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode
import os
import pandas as pd


def detect_anomaly(series, threshold, mag_window, score_window, sensitivity, detect_mode, batch_size):
    detector = SpectralResidual(series=series, threshold=threshold, mag_window=mag_window, score_window=score_window,
                                sensitivity=sensitivity, detect_mode=detect_mode, batch_size=batch_size)
    op = pd.DataFrame(detector.detect())
    return op




detect_anomaly(df, THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99, DetectMode.anomaly_only,32)

output = detect_anomaly(df,0.6, 100, SCORE_WINDOW, 99, DetectMode.anomaly_only,64)

output.to_csv('output.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt

outputdata = pd.read_csv('C:/Users/Usama/Documents/Research/anomalydetector/RKBLHWQ.csv')



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Import NumPy

# Assuming you have a DataFrame named outputdata
anomalies = outputdata[outputdata['isAnomaly'] == True]
normal_data = outputdata[outputdata['isAnomaly'] == False]



# Convert timestamp and value columns to NumPy arrays
normal_timestamp = normal_data['timestamp'].to_numpy()
normal_value = normal_data['value'].to_numpy()
anomaly_timestamp = anomalies['timestamp'].to_numpy()
anomaly_value = anomalies['value'].to_numpy()
fig, ax = plt.subplots()
# Plot the normal data points in blue
ax.plot(normal_timestamp, normal_value, marker='o', linestyle='', color='blue', label='Normal Data')
#fig, ax = plt.subplots()
# Plot the anomalies in red
ax.plot(anomaly_timestamp, anomaly_value, marker='o', linestyle='', color='red', label='Anomalies')

# Customize the plot
ax.set_xlabel('Timestamp')
ax.set_ylabel('Value')
ax.set_title('Zoomable Plot with Anomalies')
ax.legend()

# Enable zooming by selecting a region with the mouse
plt.subplots_adjust(left=0.1, bottom=0.25, right=0.9, top=0.9)
ax.set_autoscale_on(True)

# Show the plot
plt.show()


'''


import pandas as pd

dir = 'C:/Users/Usama/Documents/Research/anomalydetector/CSVData/RKBLHWQ.csv'
df = pd.read_csv(dir, skiprows=2)
df = df[['DateTimeStamp','Temp',  'Sal', 'DO_mgl', 'Turb']]
#df = df[['DateTimeStamp','Sal']]

df['DateTimeStamp'] = pd.to_datetime(df['DateTimeStamp'], format='%m/%d/%Y %H:%M')



#df.dropna(how='any', inplace=True)


# Define the date range
start_date = '2021-01-01'
end_date = '2022-12-31'

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
rise_threshold = 20  # Lower threshold for turbidity rise

# Set the minimum duration for a sustained rise (you can adjust this based on your data)
min_duration = 12  # Minimum duration in hours

# Initialize variables to track the start and end of a potential hurricane event
hurricane_start = None

# Iterate through the DataFrame to identify and label hurricane-affected data
for index, row in df.iterrows():
    turbidity = row['Turb']

    # Check if a potential hurricane event is starting
    if turbidity > rise_threshold and hurricane_start is None:
        hurricane_start = index

    # Check if a potential hurricane event is ending
    elif (
        turbidity < rise_threshold
        and hurricane_start is not None
        and (index - hurricane_start).seconds / 3600 >= min_duration
    ):
        hurricane_end = index

        # Label the data points within the hurricane event as 1
        df.loc[hurricane_start:hurricane_end, 'Hurricane_Label'] = 1

        # Reset the hurricane_start variable
        hurricane_start = None

# Now, 'Hurricane_Label' column contains 0 for normal data points and 1 for hurricane-affected data points












