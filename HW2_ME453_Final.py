#!/usr/bin/env python
# coding: utf-8

# ### ME 453 - HW2 

# In[3]:


# Package import
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


# In[5]:


# Problem 1.1
np.random.seed(99)
N =  100
n =  10
sigma = np.sqrt(0.01)
mu = 10
sample = sigma*np.random.randn(N, n) + mu
mean = np.zeros((N,)) 
cover = np.zeros((N,)) 
z_value = 1.96
CI =  np.zeros((N,2)) 
for i in range(100):
    mean[i] = np.mean(sample[i])
    margin_error = z_value * (sigma / np.sqrt(n))
    CI[i, 0]  = mean[i] - margin_error
    CI[i, 1] = mean[i] + margin_error
    cover[i] = 1 if CI[i, 0] <= mu <= CI[i, 1] else 0
    
print(CI)

coverage_rate = np.mean(cover)
print(f"Coverage Rate: {coverage_rate * 100:.2f}%")
# Note: No plot needed for Problem 1.1


# In[21]:


# Problem 1.2
Bad = np.where(cover == 0)[0] # All not covered sample
# Good: Find all "covered" sample
Good = np.where(cover == 1)[0]

plt.plot([0, 100], [mu, mu], '--', color='orange', linewidth=1, label='true mean') 
# Error bar for Bad
yerr_bad = [mean[Bad] - CI[Bad, 0], CI[Bad, 1] - mean[Bad]]
plt.errorbar(Bad, mean[Bad], yerr=yerr_bad, 
             marker='o', ms=3, color='r', ls='', ecolor='r', elinewidth=0.5, capsize=2, label='Fail')

# Error bar for Good
yerr_good = [mean[Good] - CI[Good, 0], CI[Good, 1] - mean[Good]]
plt.errorbar(Good, mean[Good], yerr=yerr_good, 
             marker='o', ms=3, color='g', ls='', ecolor='g', elinewidth=0.5, capsize=2, label='Capture Successfully')

# Necessary plot information
plt.xlabel('Sample ID')
plt.ylabel('Confidence Interval')
plt.title('Confidence Intervals for 100 Samples')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


# In[29]:


# Problem 1.3

num_failed = len(Bad)

print(f"Number of CIs that failed to capture μ: {num_failed}")

Bad = np.where(cover == 0)[0] # All not covered sample
plt.plot([0, 100], [mu, mu], '--', color='orange', linewidth=1, label='true mean') 
yerr_bad = [mean[Bad] - CI[Bad, 0], CI[Bad, 1] - mean[Bad]]
plt.errorbar(Bad, mean[Bad], yerr=yerr_bad, 
             marker='o', ms=3, color='r', ls='', ecolor='r', elinewidth=0.5, capsize=2, label='Fail')
plt.xlabel('Sample ID')
plt.ylabel('Confidence Interval')
plt.title('Error Plot of Confidence Intervals for Bad Samples')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()


# In[55]:


# Problem 2.1-3
sigma = 2
mu = 16
alpha = 0.05
delta = 16 - 20 # mean shift
sample_n = np.arange(1,11,1)
Z_alpha2 = st.norm.ppf(1- alpha/2)
beta = np.zeros((10,))


n = 2.6244
Z_beta = (delta * np.sqrt(n)) / sigma
beta_for_2_62 = st.norm.cdf(Z_alpha2 + Z_beta) - st.norm.cdf(-Z_alpha2 + Z_beta)
print(f"Beta value when n=2.62 is: {beta_for_2_62:.4f}")

n = 3
Z_beta = (delta * np.sqrt(n)) / sigma
beta_for_3 = st.norm.cdf(Z_alpha2 + Z_beta) - st.norm.cdf(-Z_alpha2 + Z_beta)
print(f"Beta value when n=3 is: {beta_for_3:.4f}")



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

#####################################################
2.2


# As n increases, the beta value deceases exponentially!!

############################################################

n = 100000000
Z_beta = (delta * np.sqrt(n)) / sigma
beta_for_infinity = st.norm.cdf(Z_alpha2 + Z_beta) - st.norm.cdf(-Z_alpha2 + Z_beta)
print(f"Beta value when n=infinty is: {beta_for_infinity:.4f}")


#####################################################
2.3

# As n increases to infinity, the beta value approaches 0. This means
# that there is a 0% chance failling to reject the null hypothesis, so the possibilities of
# error is zero. In other words, the test has perfect power to detect an effect.
#This isn't very pracitical in a real world applicaiton as more
# sample sizes take up a lot of time, effort, and money 
############################################################


# In[ ]:




