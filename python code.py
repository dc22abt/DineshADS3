import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def scaler(df):
    """ Expects a dataframe and normalises all 
        columnsto the 0-1 range. It also returns 
        dataframes with minimum and maximum for
        transforming the cluster centres"""

    # Uses the pandas methods
    df_min = df.min()
    df_max = df.max()

    df = (df-df_min) / (df_max - df_min)

    return df, df_min, df_max


def backscale(arr, df_min, df_max):
    """ Expects an array of normalised cluster centres and scales
        it back. Returns numpy array.  """

    # convert to dataframe to enable pandas operations
    minima = df_min.to_numpy()
    maxima = df_max.to_numpy()

    # loop over the "columns" of the numpy array
    for i in range(len(minima)):
        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]

    return arr



# Read CSV
file_name = 'API_EN.ATM.CO2E.PP.GD_DS2_en_csv_v2_5362895.csv'
df = pd.read_csv(file_name, skiprows=4)

# Selecting the columns to be used
columns_to_use = [str(year) for year in range(1990, 2020)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Normalize the data using the custom scaler function
df_normalized, df_min, df_max = scaler(df_years[columns_to_use])

# Find the optimal number of clusters using the elbow method
inertia = []
num_clusters = range(1, 11)

for k in num_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_normalized)
    inertia.append(kmeans.inertia_)

# Plot the explained variation as a function of the number of clusters
plt.figure(figsize=(12, 8))
plt.plot(num_clusters, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Set the optimal number of clusters based on the elbow plot
optimal_clusters = 2

# Perform K-means clustering with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df_years['Cluster'] = kmeans.fit_predict(df_normalized)

# Calculate cluster centers and scale them back to the original scale
cluster_centers = backscale(kmeans.cluster_centers_, df_min, df_max)

# Plot the clusters
plt.figure(figsize=(12, 8))

for i in range(optimal_clusters):
    cluster_data = df_years[df_years['Cluster'] == i][columns_to_use].mean()
    plt.plot(columns_to_use, cluster_data, label=f'Cluster {i+1}', linewidth=2)

plt.plot(columns_to_use, cluster_centers.T, 'k+', markersize=10, label='Cluster Centers')

plt.xlabel('Year')
plt.ylabel('CO2 emissions (kg per PPP $ of GDP)')
plt.title('CO2 Emissions Clustering')

# Adjust x-axis labels to display in intervals of 5 years
years_interval = [str(year) for year in range(1990, 2020, 5)]
plt.xticks(years_interval, years_interval)

plt.legend()
plt.show()


import random

# Create a new DataFrame to store 5 country names from each cluster
cluster_countries = pd.DataFrame()

# List of priority countries
priority_countries = ['China', 'United States', 'Russian Federation', 'Germany', 'Ukraine']

for i in range(optimal_clusters):
    cluster_data = df_years[df_years['Cluster'] == i][['Country Name', 'Cluster']]
    
    # Select priority countries if they belong to the current cluster
    priority_cluster_data = cluster_data[cluster_data['Country Name'].isin(priority_countries)].head(5)

    # Select random countries from the remaining countries in the cluster
    remaining_cluster_data = cluster_data[~cluster_data['Country Name'].isin(priority_countries)].sample(5 - len(priority_cluster_data))

    # Combine priority countries and random countries
    combined_cluster_data = pd.concat([priority_cluster_data, remaining_cluster_data])

    # Add the combined_cluster_data to the cluster_countries DataFrame
    cluster_countries = pd.concat([cluster_countries, combined_cluster_data])

# Reset the index and display the DataFrame
cluster_countries.reset_index(drop=True, inplace=True)
display(cluster_countries)


def linear_model(x, a, b):
    return a * x + b

""" Module errors. It provides the function err_ranges which calculates upper
and lower limits of the confidence interval. """

import numpy as np


def err_ranges(x, func, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p, s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper   


# Curve Fitting
from scipy.optimize import curve_fit

# Read CSV
file_name = 'API_EN.ATM.CO2E.PP.GD_DS2_en_csv_v2_5362895.csv'
df = pd.read_csv(file_name, skiprows=4)

# Selecting the columns to be used
columns_to_use = [str(year) for year in range(1990, 2020)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())


country_name = 'Germany'
data = df_years[df_years['Country Name'] == country_name][columns_to_use].values.flatten()

# Create an array of years as x values
x = np.arange(1990, 2020)

# Fit the model to the data
popt, pcov = curve_fit(linear_model, x, data)

# Calculate errors (standard deviations) for the parameters
sigma = np.sqrt(np.diag(pcov))

# Predict values for the entire range (1990-2039)
x_full_range = np.arange(1990, 2040)
y_full_range = linear_model(x_full_range, *popt)

# Calculate the lower and upper limits of the confidence range for the entire range
lower_full_range, upper_full_range = err_ranges(x_full_range, linear_model, popt, sigma)

plt.figure(figsize=(10, 6))

# Plot the original data
plt.plot(x, data, 'bo', label='Data')

# Plot the best-fitting function for the entire range
plt.plot(x_full_range, y_full_range, 'r-', label='Best-fitting function')

# Plot the confidence range for the entire range
plt.fill_between(x_full_range, lower_full_range, upper_full_range, color='gray', alpha=0.3, label='Confidence range')

# Adjust the y-axis scale
plt.yscale('log')

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('CO2 emissions (kg per PPP $ of GDP)')
plt.title(f'{country_name} CO2 Emissions')
plt.legend()

# Show the plot
plt.show()


# Plotting a comparison Line Map
file_name = 'API_EN.ATM.CO2E.PP.GD_DS2_en_csv_v2_5362895.csv'
df = pd.read_csv(file_name, skiprows=4)

# Selecting the columns to be used
columns_to_use = [str(year) for year in range(1990, 2020)]
df_years = df[['Country Name', 'Country Code'] + columns_to_use]

# Fill missing values with the mean
df_years = df_years.fillna(df_years.mean())

# Select five countries for comparison
country_list = ['United States', 'China']

plt.figure(figsize=(12, 8))

for country_name in country_list:
    country_data = df_years.loc[df_years['Country Name'] == country_name, columns_to_use].values[0]
    plt.plot(columns_to_use, country_data, label=country_name)

# Add labels and legend
plt.xlabel('Year')
plt.ylabel('CO2 emissions (kg per PPP $ of GDP)')
plt.title('CO2 Emissions Comparison')
plt.xticks(rotation=45)

# Adjust y-axis scale
plt.yscale('log')

plt.legend()
plt.show()
