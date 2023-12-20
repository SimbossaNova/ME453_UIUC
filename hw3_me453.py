import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm
import pandas as pd


n1 = 16
sigma = 1.0

def compute_B1(delta, sigma, n1):
    term1 = 3 - delta / (sigma / np.sqrt(n1))
    term2 = -3 - delta / (sigma / np.sqrt(n1))
    return norm.cdf(term1) - norm.cdf(term2)

n2=25

def compute_B2(delta, sigma, n2):
    term1 = 3.09 - delta / (sigma / np.sqrt(n2))
    term2 = -3.09 - delta / (sigma / np.sqrt(n2))
    return norm.cdf(term1) - norm.cdf(term2)


delta_values = np.linspace(0, 2, 500)  # from 0*sigma to 5*sigma
B1_values = [compute_B1(delta, sigma, n1) for delta in delta_values]
B2_values = [compute_B1(delta, sigma, n2) for delta in delta_values]


plt.figure(figsize=(10, 6))
plt.plot(delta_values, B1_values, '-r', label='Option 1')
plt.plot(delta_values, B2_values, '-b', label='Option 2')
plt.title('OC Cyrve')
plt.xlabel('Mean Shift ($\delta$)')
plt.ylabel('Type II Error (B)')
plt.grid(True)
plt.legend()
plt.show()


data = [
    [1, 4803],
    [2, 5308],
    [3, 5128],
    [4, 10612],
    [5, 6285],
    [6, 6644],
    [7, 6090],
    [8, 7861],
    [9, 6149],
    [10, 5605],
    [11, 7401],
    [12, 3281],
    [13, 5053],
    [14, 3614],
    [15, 8059],
    [16, 4112],
    [17, 5726],
    [18, 3476],
    [19, 4502],
    [20, 7113],
    [21, 5447],
    [22, 5269],
    [23, 4538],
    [24, 5435],
    [25, 5404],
    [26, 4306],
    [27, 3985],
    [28, 5060],
    [29, 2934],
    [30, 9596]
]

# Extracting the first 20 values
samples_20_data = [row[1] for row in data[:20]]
samples_30_data = [row[1] for row in data[:30]]


#print(samples_20_data)
#print(samples_30_data)


# Calculate moving range
moving_ranges = [abs(samples_20_data[i+1] - samples_20_data[i]) for i in range(len(samples_20_data) - 1)]
moving_ranges_30 = [abs(samples_30_data[i+1] - samples_30_data[i]) for i in range(len(samples_30_data) - 1)]


# Calculating average moving range (R-bar)
R_bar = sum(moving_ranges) / len(moving_ranges)

# Constants for n=2
D3 = 0
D4 = 3.267

# Control limits for moving range
UCL_MR = D4 * R_bar
LCL_MR = D3 * R_bar

# Control limits for individual values
UCL_X = np.mean(samples_20_data) + 3 * (R_bar/1.128)
LCL_X = np.mean(samples_20_data) - 3 * (R_bar/1.128)

# Correcting the plotting for the first 20 samples
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
for ax in axes:
    ax.set_xlim(1, 22) 

# Individual value control chart
axes[0].plot(range(1, 21), samples_20_data, marker="o", color="b", linestyle="-", label='Individual Values')
axes[0].axhline(y=UCL_X, color="r", linestyle="--", label='UCL')
axes[0].axhline(y=np.mean(samples_20_data), color="g", linestyle="-", label='CL')
axes[0].axhline(y=LCL_X, color="r", linestyle="--", label='LCL')
axes[0].set_title("Individual Value Control Chart Phase I")
axes[0].set_xlabel("Sample Number")
axes[0].set_ylabel("Sample Value")
axes[0].set_ylim(0, 12500)
axes[0].set_xticks(range(1, 21))
axes[0].grid(True)
axes[0].legend()

axes[0].annotate(f'UCL: {UCL_X:.2f}', xy=(22, UCL_X), xytext=(22, UCL_X + 200), textcoords='data', va='center')
axes[0].annotate(f'CL: {np.mean(samples_20_data):.2f}', xy=(22, np.mean(samples_20_data)), xytext=(22, np.mean(samples_20_data) + 200), textcoords='data', va='center')
axes[0].annotate(f'LCL: {LCL_X:.2f}', xy=(22, LCL_X), xytext=(22, LCL_X + 200), textcoords='data', va='center')



# Moving range control chart
axes[1].plot(range(2, 21), moving_ranges, marker="o", color="b", linestyle="-",label='Moving Range Values')  # Starting from 2 since MR starts from the second sample
axes[1].axhline(y=UCL_MR, color="r", linestyle="--", label='UCL')
axes[1].axhline(y=R_bar, color="g", linestyle="-", label='CL')
axes[1].axhline(y=LCL_MR, color="r", linestyle="--", label='LCL')
axes[1].set_title("Moving Range Control Chart Phase I")
axes[1].set_xlabel("Sample Number")
axes[1].set_ylabel("Moving Range")
axes[1].set_ylim(0, 8000)
axes[1].set_xticks(range(1, 21))
axes[1].grid(True)
axes[1].legend()
axes[1].annotate(f'UCL: {UCL_MR:.2f}', xy=(22, UCL_MR), xytext=(22, UCL_MR + 100), textcoords='data', va='center')
axes[1].annotate(f'CL: {R_bar:.2f}', xy=(22, R_bar), xytext=(22, R_bar + 100), textcoords='data', va='center')
axes[1].annotate(f'LCL: {LCL_MR:.2f}', xy=(22, LCL_MR), xytext=(22, LCL_MR + 100), textcoords='data', va='center')

plt.tight_layout()
plt.show()





fig, axes = plt.subplots(2, 1, figsize=(10, 10))
for ax in axes:
    ax.set_xlim(1, 33)



axes[0].plot(range(1, 31), samples_30_data, marker="o", color="b", linestyle="-", label='Individual Values')
axes[0].axhline(y=UCL_X, color="r", linestyle="--", label='UCL')
axes[0].axhline(y=np.mean(samples_20_data), color="g", linestyle="-",label='CL')
axes[0].axhline(y=LCL_X, color="r", linestyle="--", label='LCL')
axes[0].axvline(x=20, color="b", linestyle=":", linewidth=2)  # Vertical dotted line
axes[0].set_ylim(0, 12500)
axes[0].set_title("Individual Value Control Chart Phase II")
axes[0].set_xlabel("Sample Number")
axes[0].set_ylabel("Sample Value")
axes[0].set_xticks(range(1, 31))
axes[0].grid(True)
axes[0].legend()
axes[0].annotate(f'UCL: {UCL_X:.2f}', xy=(32, UCL_X), xytext=(32, UCL_X + 200), textcoords='data', va='center')
axes[0].annotate(f'CL: {np.mean(samples_20_data):.2f}', xy=(32, np.mean(samples_20_data)), xytext=(32, np.mean(samples_20_data) + 200), textcoords='data', va='center')
axes[0].annotate(f'LCL: {LCL_X:.2f}', xy=(32, LCL_X), xytext=(32, LCL_X + 200), textcoords='data', va='center')




axes[1].plot(range(2, 31), moving_ranges_30, marker="o", color="b", linestyle="-", label='Moving Range Values')  # Starting from 2 since MR starts from the second sample
axes[1].axhline(y=UCL_MR, color="r", linestyle="--", label='UCL')
axes[1].axhline(y=R_bar, color="g", linestyle="-", label='CL')
axes[1].axhline(y=LCL_MR, color="r", linestyle="--", label='LCL')
axes[1].axvline(x=20, color="b", linestyle=":", linewidth=2)  # Vertical dotted line
axes[1].set_ylim(0, 8000)
axes[1].set_title("Moving Range Control Chart Phase II")
axes[1].set_xlabel("Sample Number")
axes[1].set_ylabel("Moving Range")
axes[1].set_xticks(range(1, 31))
axes[1].grid(True)
axes[1].legend()
axes[1].annotate(f'UCL: {UCL_MR:.2f}', xy=(32, UCL_MR), xytext=(32, UCL_MR + 100), textcoords='data', va='center')
axes[1].annotate(f'CL: {R_bar:.2f}', xy=(32, R_bar), xytext=(32, R_bar + 100), textcoords='data', va='center')
axes[1].annotate(f'LCL: {LCL_MR:.2f}', xy=(32, LCL_MR), xytext=(32, LCL_MR + 100), textcoords='data', va='center')



plt.tight_layout()
plt.show()


n = 6
total_X_bar = 6000
total_R = 150
D4 = 2.004
d2 = 2.534

# Calculate average of sample means and average range
X_bar_avg = total_X_bar / 30
R_avg = total_R / 30

# Updated control limits for X_bar
UCL_X_bar_updated = 202.415
LCL_X_bar_updated = 197.56

# Estimate process standard deviation
sigma_hat = R_avg / d2
sigma_X_bar = sigma_hat / np.sqrt(n)

# Task 4: Simulation Study on β-risk

# (a) Generate 1000 samples with sample size n=6 using X ~ N(199, sigma_hat^2)
samples_1000 = np.random.normal(199, sigma_hat, (1000, 6))

# (b) Compute X_bar for each sample and check if it falls within the control limits
X_bars_1000 = np.mean(samples_1000, axis=1)
in_control = np.logical_and(X_bars_1000 >= LCL_X_bar_updated, X_bars_1000 <= UCL_X_bar_updated)

# (c) Count how many samples fall within the control limits and calculate the beta error rate
num_in_control = np.sum(in_control)
print(num_in_control)
beta_error_rate_simulation = 1 - (num_in_control / 1000)

print(beta_error_rate_simulation)



A3 = 1.427
B4 = 2.089
B3 = 0
c4 = 0.9400

sample1data_str = """
1.3235 1.4128 1.6744 1.4573 1.6914 
1.4314 1.3592 1.6075 1.4666 1.6109 
1.4284 1.4871 1.4932 1.4324 1.5674 
1.5028 1.6352 1.3841 1.2831 1.5507 
1.5604 1.2735 1.5265 1.4363 1.6441 
1.5955 1.5451 1.3574 1.3281 1.4198 
1.6274 1.5064 1.8366 1.4177 1.5144 
1.419 1.4303 1.6637 1.6067 1.5519 
1.3884 1.7277 1.5355 1.5176 1.3688
1.4039 1.6697 1.5089 1.4627 1.522 
1.4158 1.7667 1.4278 1.5928 1.4181
1.5821 1.3355 1.5777 1.3908 1.7559 
1.2856 1.4106 1.4447 1.6398 1.1928 
1.4951 1.4036 1.5893 1.6458 1.4969 
1.3589 1.2863 1.5996 1.2497 1.5471 
1.5747 1.5301 1.5171 1.1839 1.8662 
1.368 1.7269 1.3957 1.5014 1.4449 
1.4163 1.3864 1.3057 1.621 1.5573 
1.5796 1.4185 1.6541 1.5116 1.7247 
1.7106 1.4412 1.2361 1.382 1.7601 
1.4371 1.5051 1.3485 1.567 1.488 
1.4738 1.5936 1.6583 1.4973 1.472 
1.5917 1.4333 1.5551 1.5295 1.6866 
1.6399 1.5243 1.5705 1.5563 1.553 
1.5797 1.3663 1.624 1.3732 1.6887 

"""

# Splitting the string by lines and then by spaces to get the matrix
sample1data = [list(map(float, line.split())) for line in sample1data_str.strip().split("\n")]

# Convert the parsed data into a DataFrame
data_df = pd.DataFrame(sample1data)
print(data_df.head())
print(data_df.shape)

data_df['X_bar'] = data_df.mean(axis=1)
print(data_df['X_bar'])
std_values = [
    0.163495434,
0.111110472,
0.056518316,
0.138908988,
0.141221663,
0.116787529,
0.161361922,
0.107710222,
0.143871905,
0.098849168,
0.15476832,`
0.168236887,
0.169921482,
0.093731281,
0.156795606,
0.242323833,
0.143202434,
0.128926386,
0.119544364,
0.222962452,
0.081862647,
0.083205439,
0.092245206,
0.043140005,
0.148163919

]

# Create a DataFrame
std_df = pd.DataFrame(std_values, columns=['S Deviation'])

# Attach the standard deviations to your data DataFrame
data_df['S'] = std_df['S Deviation']

# Your code for calculating X_bar and S_bar...
X_bar_bar = data_df['X_bar'].mean()
S_bar = data_df['S'].mean()
print(S_bar)

# Your control limit calculations...
UCL_X = X_bar_bar + A3 * S_bar
LCL_X = X_bar_bar - A3 * S_bar

UCL_S = B4 * S_bar
LCL_S = B3 * S_bar

# Your plotting code...
fig, axes = plt.subplots(2, 1, figsize=(10, 10))

for ax in axes:
    ax.set_xlim(1, 28)  # Adjust xlim to 28 to add some whitespace to the right

# Individual value control chart
axes[0].plot(range(1, 26), data_df['X_bar'], marker="o", color="b", linestyle="-", label='Sample Mean')
axes[0].axhline(y=UCL_X, color="r", linestyle="--", label='UCL')
axes[0].axhline(y=X_bar_bar, color="g", linestyle="-", label='Average Sample Mean')
axes[0].axhline(y=LCL_X, color="r", linestyle="--", label='LCL')
axes[0].set_title("Xbar Chart Phase I")
axes[0].set_xlabel("Sample Number")
axes[0].set_ylabel("Sample Mean")
axes[0].set_xticks(range(1, 26))
axes[0].grid(True)
axes[0].legend(loc='upper right', bbox_to_anchor=(1, 0.9))  # Lowered legend slightly

# Annotations for Xbar Chart
axes[0].annotate(f'UCL: {UCL_X:.2f}', xy=(26, UCL_X), xytext=(26, UCL_X + 0.02), textcoords='data', va='center')
axes[0].annotate(f'CL: {X_bar_bar:.2f}', xy=(26, X_bar_bar), xytext=(26, X_bar_bar + 0.02), textcoords='data', va='center')
axes[0].annotate(f'LCL: {LCL_X:.2f}', xy=(26, LCL_X), xytext=(26, LCL_X + 0.02), textcoords='data', va='center')

# S chart
axes[1].plot(range(1, 26), data_df['S'], marker="o", color="b", linestyle="-", label='Sample Std Dev')
axes[1].axhline(y=UCL_S, color="r", linestyle="--", label='UCL')
axes[1].axhline(y=S_bar, color="g", linestyle="-", label='Average Sample Std Dev')
axes[1].axhline(y=LCL_S, color="r", linestyle="--", label='LCL')
axes[1].set_title("S Chart Phase I")
axes[1].set_xlabel("Sample Number")
axes[1].set_ylabel("Sample Standard Deviation")
axes[1].set_xticks(range(1, 26))
axes[1].grid(True)
axes[1].legend(loc='upper right', bbox_to_anchor=(1, 0.9))  # Lowered legend slightly

# Annotations for S Chart
axes[1].annotate(f'UCL: {UCL_S:.2f}', xy=(26, UCL_S), xytext=(26, UCL_S + 0.02), textcoords='data', va='center')
axes[1].annotate(f'CL: {S_bar:.2f}', xy=(26, S_bar), xytext=(26, S_bar + 0.02), textcoords='data', va='center')
axes[1].annotate(f'LCL: {LCL_S:.2f}', xy=(26, LCL_S), xytext=(26, LCL_S + 0.02), textcoords='data', va='center')

plt.tight_layout()
plt.show()

sample2data_str ="""
1.3235 1.4128 1.6744 1.4573 1.6914 
1.4314 1.3592 1.6075 1.4666 1.6109 
1.4284 1.4871 1.4932 1.4324 1.5674 
1.5028 1.6352 1.3841 1.2831 1.5507 
1.5604 1.2735 1.5265 1.4363 1.6441 
1.5955 1.5451 1.3574 1.3281 1.4198 
1.6274 1.5064 1.8366 1.4177 1.5144 
1.419 1.4303 1.6637 1.6067 1.5519 
1.3884 1.7277 1.5355 1.5176 1.3688 
1.4039 1.6697 1.5089 1.4627 1.522 
1.4158 1.7667 1.4278 1.5928 1.4181 
1.5821 1.3355 1.5777 1.3908 1.7559 
1.2856 1.4106 1.4447 1.6398 1.1928 
1.4951 1.4036 1.5893 1.6458 1.4969 
1.3589 1.2863 1.5996 1.2497 1.5471 
1.5747 1.5301 1.5171 1.1839 1.8662 
1.368 1.7269 1.3957 1.5014 1.4449 
1.4163 1.3864 1.3057 1.621 1.5573 
1.5796 1.4185 1.6541 1.5116 1.7247 
1.7106 1.4412 1.2361 1.382 1.7601 
1.4371 1.5051 1.3485 1.567 1.488 
1.4738 1.5936 1.6583 1.4973 1.472 
1.5917 1.4333 1.5551 1.5295 1.6866 
1.6399 1.5243 1.5705 1.5563 1.553 
1.5797 1.3663 1.624 1.3732 1.6887 
1.4483 1.5458 1.4538 1.4303 1.6206 
1.5435 1.6899 1.583 1.3358 1.4187 
1.5175 1.3446 1.4723 1.6657 1.6661 
1.5454 1.0931 1.4072 1.5039 1.5264 
1.4418 1.5059 1.5124 1.462 1.6263 
1.4301 1.2725 1.5945 1.5397 1.5252 
1.4981 1.4506 1.6174 1.5837 1.4962 
1.3009 1.506 1.6231 1.5831 1.6454 
1.4132 1.4603 1.5808 1.7111 1.7313 
1.3817 1.3135 1.4953 1.4894 1.4596 
1.5765 1.7014 1.4026 1.2773 1.4541 
1.4936 1.4373 1.5139 1.4808 1.5293 
1.5729 1.6738 1.5048 1.5651 1.7473 
1.8089 1.5513 1.825 1.4389 1.6558 
1.6236 1.5393 1.6738 1.8698 1.5036 
1.412 1.7931 1.7345 1.6391 1.7791 
1.7372 1.5663 1.491 1.7809 1.5504 
1.5971 1.7394 1.6832 1.6677 1.7974
1.4295 1.6536 1.9134 1.7272 1.437 
1.6217 1.822 1.7915 1.6744 1.9404 


"""

sample2data = [list(map(float, line.split())) for line in sample2data_str.strip().split("\n")]

# Convert the parsed data into a DataFrame
data_df_2 = pd.DataFrame(sample2data)
print(data_df_2.head())
print(data_df_2.shape)

data_df_2['X_bar_tot'] = data_df_2.mean(axis=1)
print(data_df_2['X_bar_tot'])

std_values_2 = [
    0.163495434,
0.111110472,
0.056518316,
0.138908988,
0.141221663,
0.116787529,
0.161361922,
0.107710222,
0.143871905,
0.098849168,
0.15476832,
0.168236887,
0.169921482,
0.093731281,
0.156795606,
0.242323833,
0.143202434,
0.128926386,
0.119544364,
0.222962452,
0.081862647,
0.083205439,
0.092245206,
0.043140005,
0.148163919,
0.081097367,
0.13911257,
0.136696664,
0.187748222,
0.071594322,
0.126466241,
0.068890602,
0.139510519,
0.143376857,
0.07834491,
0.162827169,
0.035305056,
0.0966211,
0.165856827,
0.144000424,
0.15710133,
0.126353049,
0.075675049,
0.204784467,
0.125831892

]

# Create a DataFrame
std_df_2 = pd.DataFrame(std_values_2, columns=['S Deviation'])
data_df_2['S'] = std_df_2['S Deviation']

print(std_df_2)

# Your plotting code...
fig, axes = plt.subplots(2, 1, figsize=(20, 10))

for ax in axes:
    ax.set_xlim(1, 46)  # Adjust xlim to 28 to add some whitespace to the right

# Individual value control chart
axes[0].plot(range(1, 46), data_df_2['X_bar_tot'], marker="o", color="b", linestyle="-", label='Sample Mean')
axes[0].axhline(y=UCL_X, color="r", linestyle="--", label='UCL')
axes[0].axhline(y=X_bar_bar, color="g", linestyle="-", label='Average Sample Mean')
axes[0].axhline(y=LCL_X, color="r", linestyle="--", label='LCL')
axes[0].axvline(x=25, color="b", linestyle=":", linewidth=2)  # Vertical dotted line

axes[0].set_title("Xbar Chart Phase II")
axes[0].set_xlabel("Sample Number")
axes[0].set_ylabel("Sample Mean")
axes[0].set_xticks(range(1, 48))
axes[0].grid(True)
axes[0].legend(loc='lower left', bbox_to_anchor=(1, 0.9))  # Lowered legend slightly

# Annotations for Xbar Chart
axes[0].annotate(f'UCL: {UCL_X:.2f}', xy=(47, UCL_X), xytext=(47, UCL_X + 0.02), textcoords='data', va='center')
axes[0].annotate(f'CL: {X_bar_bar:.2f}', xy=(47, X_bar_bar), xytext=(47, X_bar_bar + 0.02), textcoords='data', va='center')
axes[0].annotate(f'LCL: {LCL_X:.2f}', xy=(47, LCL_X), xytext=(47, LCL_X + 0.02), textcoords='data', va='center')

# S chart
axes[1].plot(range(1, 46), data_df_2['S'], marker="o", color="b", linestyle="-", label='Sample Std Dev')
axes[1].axhline(y=UCL_S, color="r", linestyle="--", label='UCL')
axes[1].axhline(y=S_bar, color="g", linestyle="-", label='Average Sample Std Dev')
axes[1].axhline(y=LCL_S, color="r", linestyle="--", label='LCL')
axes[1].axvline(x=25, color="b", linestyle=":", linewidth=2)  # Vertical dotted line

axes[1].set_title("S Chart Phase II")
axes[1].set_xlabel("Sample Number")
axes[1].set_ylabel("Sample Standard Deviation")

axes[1].set_xticks(range(1, 48))
axes[1].grid(True)
axes[1].legend(loc='upper left', bbox_to_anchor=(1, 0.9))  # Lowered legend slightly

# Annotations for S Chart
axes[1].annotate(f'UCL: {UCL_S:.2f}', xy=(47, UCL_S), xytext=(47, UCL_S + 0.02), textcoords='data', va='center')
axes[1].annotate(f'CL: {S_bar:.2f}', xy=(47, S_bar), xytext=(47, S_bar + 0.02), textcoords='data', va='center')
axes[1].annotate(f'LCL: {LCL_S:.2f}', xy=(47, LCL_S), xytext=(47, LCL_S + 0.02), textcoords='data', va='center')

plt.tight_layout()
plt.show()