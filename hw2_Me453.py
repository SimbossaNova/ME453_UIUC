import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import norm



sigma = 2
mu = 16
alpha = 0.05
delta = 16 - 20 # mean shift
sample_n = np.arange(1,11,1)
Z_alpha2 = st.norm.ppf(1- alpha/2)
beta = np.zeros((10,))

print(f"Beta value when n=3 is: {beta[2]:.4f}")
n = 2.62
Z_beta = (delta * np.sqrt(n)) / sigma
beta_for_2_62 = st.norm.cdf(Z_alpha2 + Z_beta) - st.norm.cdf(-Z_alpha2 + Z_beta)
print(f"Beta value when n=2.62 is: {beta_for_2_62:.4f}")


for i, n in enumerate(sample_n):
    Z_beta = (delta * np.sqrt(n)) / sigma
    beta[i] = st.norm.cdf(Z_alpha2 + Z_beta) - st.norm.cdf(-Z_alpha2 + Z_beta)

# Use a plot to show the relation between sample_n and beta
plt.plot(sample_n, beta, '-o', color='blue')
plt.axhline(0.1, color='red', linestyle='--')
plt.xlabel('Sample Size (n)')
plt.ylabel('Type II Error (β)')
plt.title('Type II Error vs. Sample Size')
plt.legend(['Beta', 'Desired Beta=0.1'])
plt.grid(True)
plt.show()



## Given values
n = 4
sigma = 1.0

# Define the B function for type II error
def compute_B(delta, sigma, n):
    term1 = 3.09 - delta / (sigma / np.sqrt(n))
    term2 = -3.09 - delta / (sigma / np.sqrt(n))
    return norm.cdf(term1) - norm.cdf(term2)

# Generate delta values
delta_values = np.linspace(0, 5, 500)  # from 0*sigma to 5*sigma

# Compute B for each delta
B_values = [compute_B(delta, sigma, n) for delta in delta_values]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(delta_values, B_values, '-r', label='Type II Error vs. Mean Shift')
plt.title('Type II Error (B) vs. Mean Shift ($\delta$)')
plt.xlabel('Mean Shift ($\delta$)')
plt.ylabel('Type II Error (B)')
plt.grid(True)
plt.legend()
plt.show()







n = 9
sigma = 1.0

def compute_B1(delta, sigma, n):
    term1 = 2.33 - delta / (sigma / np.sqrt(n))
    term2 = -2.33 - delta / (sigma / np.sqrt(n))
    return norm.cdf(term1) - norm.cdf(term2)

def compute_B2(delta, sigma, n):
    Pl = norm.cdf(-delta / (sigma / np.sqrt(n)))
    Pu = 1 - Pl
    return 1 - Pl**4 - Pu**4

delta_values = np.linspace(0, 2, 500)  # from 0*sigma to 5*sigma

# Compute B for each delta using both rules
B1_values = [compute_B1(delta, sigma, n) for delta in delta_values]
B2_values = [compute_B2(delta, sigma, n) for delta in delta_values]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(delta_values, B1_values, '-r', label='Rule 1')
plt.plot(delta_values, B2_values, '-b', label='Rule 2')
plt.title('Type II Error (B) vs. Mean Shift ($\delta$)')
plt.xlabel('Mean Shift ($\delta$)')
plt.ylabel('Type II Error (B)')
plt.grid(True)
plt.legend()
plt.show()

