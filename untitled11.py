# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:19:08 2024

@author: mi0025
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

# Read the Excel file
file_path = 'C:/Users/mi0025/Downloads/combined_cleaned_data_cleaned.csv'
df = pd.read_excel(file_path)

# Classify columns according to themes
theme_1_columns = ['H_mental', 'H_dia', 'H_health', 'H_hypertension', 'H_NoInsurance','H_kidney']
theme_2_columns = ['ForeignBornPercentage', 'MinorityPercentage', 'MedianAge', 'PerCapitaIncome', 'PercentWithoutDiploma','UnemploymentPercentage','PercentBelowPovertyLevel','D_fertility','pop18','D_disability','pop65','pop18_64']
theme_3_columns = ['HH_one_or_more_u18', 'LackingCompletePlumbingFacilities', 'RenterOccupied', 'CrowdedHouseholdPercentage', 'HousesBuiltBefore2000','Single Parent Household']

# Ensure the 'COUNTY' column is included
county_column = ['CountyState']

# Function to normalize data
def normalize(df, columns):
    scaler = MinMaxScaler()
    df.loc[:, columns] = scaler.fit_transform(df.loc[:, columns])
    return df

# Function to perform PCA and return L2 norms
def perform_pca_and_get_l2_norms(df, columns, n_components=3):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(df[columns])
    pca_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    l2_norms = np.linalg.norm(pca_df.values, axis=1)
    return l2_norms

# Function to perform Pareto ranking with modified dominance criteria
def pareto_ranking(norms_df, columns):
    # Initialize ranking dictionary
    ranks = {}
    rank = 1
    remaining_counties = norms_df[['CountyState'] + columns].copy()
    
    while not remaining_counties.empty:
        pareto_front = []
        
        for i, row in remaining_counties.iterrows():
            county_dominated = False
            
            for j, other_row in remaining_counties.iterrows():
                if i != j:
                    if all(other_row[col] >= row[col] for col in columns) and any(other_row[col] > row[col] for col in columns):
                        county_dominated = True
                        break
            
            if not county_dominated:
                pareto_front.append(row)
        
        for row in pareto_front:
            ranks[row['CountyState']] = (rank, *row[columns].values)
            remaining_counties = remaining_counties[remaining_counties['CountyState'] != row['CountyState']]
        
        rank += 1
    
    return ranks

# Normalize and perform PCA for each theme
theme_1_df = df[county_column + theme_1_columns].copy()
theme_2_df = df[county_column + theme_2_columns].copy()
theme_3_df = df[county_column + theme_3_columns].copy()

theme_1_df = normalize(theme_1_df, theme_1_columns)
theme_2_df = normalize(theme_2_df, theme_2_columns)
theme_3_df = normalize(theme_3_df, theme_3_columns)

theme_1_l2_norms = perform_pca_and_get_l2_norms(theme_1_df, theme_1_columns)
theme_2_l2_norms = perform_pca_and_get_l2_norms(theme_2_df, theme_2_columns)
theme_3_l2_norms = perform_pca_and_get_l2_norms(theme_3_df, theme_3_columns)

# Combine L2 norms into a single DataFrame
l2_norms_df = pd.DataFrame({
    'CountyState': df['CountyState'],
    'Theme_1_L2': theme_1_l2_norms,
    'Theme_2_L2': theme_2_l2_norms,
    'Theme_3_L2': theme_3_l2_norms
})

# Get the Pareto rankings with modified dominance criteria
pareto_ranks_all = pareto_ranking(l2_norms_df, ['Theme_1_L2', 'Theme_2_L2', 'Theme_3_L2'])

# Convert the ranks dictionary to DataFrame for better readability
pareto_ranks_all_df = pd.DataFrame([(county, rank) + tuple(values) for county, (rank, *values) in pareto_ranks_all.items()],
                                   columns=['CountyState', 'Rank', 'Theme_1_L2', 'Theme_2_L2', 'Theme_3_L2']).sort_values(by='Rank')

# Reverse the ranks
pareto_ranks_all_df['Reversed Rank'] = pareto_ranks_all_df['Rank'].max() + 1 - pareto_ranks_all_df['Rank']

# Normalize the reversed ranks from 0 to 1 for the "All Themes" class
scaler = MinMaxScaler()
pareto_ranks_all_df['Normalized Reversed Rank'] = scaler.fit_transform(pareto_ranks_all_df[['Reversed Rank']])

# Save the output as an Excel file
output_file_path = 'normalized_pareto_ranks_all_themes.xlsx'
pareto_ranks_all_df[['CountyState', 'Rank', 'Reversed Rank', 'Normalized Reversed Rank']].to_excel(output_file_path, index=False)

# Print the normalized ranks DataFrame
print("All Themes Pareto Ranking with Normalized Reversed Ranks:")
print(pareto_ranks_all_df[['CountyState', 'Rank', 'Reversed Rank', 'Normalized Reversed Rank']])