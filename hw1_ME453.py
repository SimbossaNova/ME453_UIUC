# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:01:12 2023

@author: jacob
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import pandas as pd
from scipy.stats import ttest_ind

data = [
    [127, 124, 121, 118],
    [125, 123, 136, 131],
    [131, 120, 140, 125],
    [124, 119, 137, 133],
    [129, 128, 125, 141],
    [121, 133, 124, 125],
    [142, 137, 128, 140],
    [151, 124, 129, 131],
    [160, 142, 130, 129],
    [125, 123, 122, 126]
]

flat_data = [item for sublist in data for item in sublist]
sample_average = sum(flat_data) / len(flat_data)

print("Sample Average:", sample_average)

sample_mean = sum(flat_data) / len(flat_data)

squared_diff = [(x - sample_mean) ** 2 for x in flat_data]

variance = sum(squared_diff) / (len(flat_data) - 1)

std_deviation = math.sqrt(variance)

print("Sample Standard Deviation:", std_deviation)

# Determine the number of bins using the square root rule of thumb
num_bins = int(math.sqrt(len(flat_data)))

# Create a histogram
plt.hist(flat_data, bins=num_bins, edgecolor='black')

# Add labels and title
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of the Dataset')

# Display the histogram
plt.show()

median = np.median(flat_data)
lower_quartile = np.percentile(flat_data, 25)
upper_quartile = np.percentile(flat_data, 75)

print("Sample Median:", median)
print("Lower Quartile:", lower_quartile)
print("Upper Quartile:", upper_quartile)

# Create a time-series plot
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.xlabel('Counts')
plt.ylabel('Time in Hours')
plt.title('Time-Series Plot of Test Data')
plt.legend(['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4'])
plt.grid(True)
plt.show()

# Calculate the quartiles and IQR
q1 = np.percentile(flat_data, 25)
q3 = np.percentile(flat_data, 75)
iqr = q3 - q1

# Define the lower and upper bounds for outliers
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# Identify outliers
outliers = [value for value in flat_data if value < lower_bound or value > upper_bound]

# Flatten the nested list
flat_data = [item for sublist in data for item in sublist]

n = len(data)
X_bar = np.mean(flat_data)
Xd = np.std(flat_data, ddof=1)
CI = st.norm.ppf(1 - 0.025) * Xd 

f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(12, 6))
a0.plot(flat_data, 'b.')
a0.set_xlabel('Sample Index')
a0.set_ylabel('Failure Time (Hours)')  # Corrected ylabel
a0.grid()
a1.boxplot(flat_data)
plt.show()

print("Lower bound: ", lower_bound)
print("Upper bound: ", upper_bound)

print("Outliers:", outliers)


##############################################################
welding_data = pd.read_csv('C:\\Users\\jacob\\welding_data-1.csv')

# Count the number of welds
num_welds = welding_data.shape[0]
print("Number of welds:", num_welds)

num_defective = welding_data[welding_data["Quality"] == 1].shape[0]
defective_rate = num_defective / num_welds
defective_rate = defective_rate *100
print("Defective rate: ", defective_rate, "%")

# Construct a table of Pearson’s correlation coefficient for each pair of features
correlation_table = welding_data.corr(method = 'pearson')

# Excluding the 'Quality' column as it's not a feature for this particular task
correlation_table = correlation_table.drop(columns="Quality", index="Quality")
print(correlation_table)

plt.figure(figsize=(10, 6))
plt.scatter(welding_data["Feature3"], welding_data["Feature4"], alpha=0.5, edgecolor='k')
plt.xlabel("Feature3")
plt.ylabel("Feature4")
plt.title("Scatter plot of Feature3 vs. Feature4")
plt.grid(True)
plt.show()

features_continuous = welding_data.columns.difference(["Quality", "Configuration"])

# Calculate number of bins using the rule of thumb
num_bins = int(np.sqrt(welding_data.shape[0]))

# Plot histograms for continuous features with labeled axes using the rule of thumb for bin size
fig, axes = plt.subplots(len(features_continuous), 1, figsize=(10, 12))
fig.tight_layout(pad=5.0)

for ax, feature in zip(axes, features_continuous):
    ax.hist(welding_data[feature], bins=num_bins, edgecolor='black')
    ax.set_title(f'Histogram for {feature}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

plt.show()


plt.figure(figsize=(15, 10))
colors = ['blue', 'green', 'red', 'purple']

for i, feature in enumerate(features_continuous):
    # Plot good welds
    plt.scatter(welding_data.index[welding_data["Quality"] == 0], 
                welding_data[feature][welding_data["Quality"] == 0], 
                label=f"{feature} (Good)", s=10, color=colors[i])
    
    # Plot defective welds
    plt.scatter(welding_data.index[welding_data["Quality"] == 1], 
                welding_data[feature][welding_data["Quality"] == 1], 
                marker="x", label=f"{feature} (Defective)", s=10, color=colors[i])

plt.xlabel("Time (index)")
plt.ylabel("Feature Value")
plt.title("Time-series graph of features distinguishing good and defective joints")
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.show()

fig, axes = plt.subplots(len(features_continuous), 1, figsize=(15, 20))
fig.tight_layout(pad=5.0)

for ax, feature in enumerate(features_continuous):
    # Plot good welds
    axes[ax].scatter(welding_data.index[welding_data["Quality"] == 0], 
                     welding_data[feature][welding_data["Quality"] == 0], 
                     label="Good", s=20, color='blue', alpha=0.6)
    
    # Plot defective welds
    axes[ax].scatter(welding_data.index[welding_data["Quality"] == 1], 
                     welding_data[feature][welding_data["Quality"] == 1], 
                     marker="x", label="Defective", s=30, color='red', alpha=0.8)
    
    axes[ax].set_title(f"Time-series graph for {feature}")
    axes[ax].set_xlabel("Time (index)")
    axes[ax].set_ylabel(f"{feature} Value")
    axes[ax].legend()
    axes[ax].grid(True)

plt.show()

good_welds = welding_data[welding_data["Quality"] == 0]
defective_welds = welding_data[welding_data["Quality"] == 1]

# Calculate statistics for each group
stats_good = good_welds[features_continuous].agg(['mean', 'median', 'min', 'max', 'var'])
stats_defective = defective_welds[features_continuous].agg(['mean', 'median', 'min', 'max', 'var'])

# Calculate range for each group
stats_good.loc['range'] = stats_good.loc['max'] - stats_good.loc['min']
stats_defective.loc['range'] = stats_defective.loc['max'] - stats_defective.loc['min']

print(stats_good)
print(stats_defective)

t_test_results = {}

for feature in features_continuous:
    t_stat, p_value = ttest_ind(good_welds[feature], defective_welds[feature], equal_var=False)
    t_test_results[feature] = {'t_stat': t_stat, 'p_value': p_value}

t_test_results_df = pd.DataFrame(t_test_results).T
print(t_test_results_df)

config_counts = welding_data["Configuration"].value_counts()
config_percentages = welding_data["Configuration"].value_counts(normalize=True) * 100

config_stats_df = pd.DataFrame({
    "Counts": config_counts,
    "Percentages (%)": config_percentages
})

print(config_stats_df)

defective_rates = welding_data.groupby("Configuration")["Quality"].mean()

defective_rates_df = pd.DataFrame({
    "Defective Rate (%)": defective_rates * 100
})

print(defective_rates_df)