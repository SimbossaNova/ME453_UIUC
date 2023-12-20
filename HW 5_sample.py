#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Plot parameter setting 
plt.style.use('default')
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.figsize'] = 8, 6
plt.rcParams['figure.dpi'] = 100


# Sample code for Eigenvalues and Eigenvectors

# In[292]:


from numpy.linalg import eig
a = np.array([[16, 5], 
              [5, 9]])
w,v= np.linalg.eig(a)
print('E-value:', w)
print('E-vector', v)


# In[289]:


# 1.2 Projection W

W = v.T
# Print the projection matrix 'W'
print('Projection matrix W:', W)


# In[293]:


# 1.3 
X = [[5], [4]]
#print(x)


print(W@X)


# In[294]:


# 1.4 

total_variance = 18.60327781  + 6.39672219
PC1_var = (18.60327781 / total_variance ) *100
PC2_var = (6.39672219 /  total_variance) * 100

print(PC1_var,"%" )
print(PC2_var,"%" )


# The following example use the iris data set to demonstrate how to perform PCA, plot scree plot, and visualize data. The code is fulling functional, so it should give you a clear view of how to do it.

# In[134]:


# Loading the data
iris = datasets.load_iris()
X = iris.data
y = iris.target
X # X is has 4 featurs and 150 samples
y


# In[105]:


# Standardize the data
from sklearn.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X)

L = 4 # Plot L PCs in the scree plot
alpha = 0.1 # alpha value, used for explained variance ratio criteria

# Perform PCA
from sklearn import decomposition
pca = decomposition.PCA(n_components = L)
pca.fit(X_std) # Fit in the data
PC = pca.transform(X_std) # Project the data to new plane
var_exp = pca.explained_variance_ratio_ # Get explaind variance ratio
cum_var_exp = np.cumsum(var_exp) # Get the sum of explained variace ratio

# Scree plot
plt.bar(range(L), var_exp, alpha=0.3, align='center', label='individual explained variance')
plt.plot(np.arange(4), cum_var_exp, ".-",label='cumulative explained variance')
plt.plot([0, L], [1-alpha, 1-alpha], label="97%")
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.grid()
plt.legend(loc='best')


# In[106]:


# 3D scatter plot
col = ["blue", "red", "green"] # Use col to automatic select color for different class
fig = plt.figure(figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=120)
for i in range(len(PC)):
    ax.scatter(PC[i, 0], PC[i, 1], PC[i, 2], c=col[y[i]])

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")


# Use the sample code for iris data set as reference and finish the code below to analyze "bearing" data. "bearing.csv" act as X and "wearlevel.csv" act as y.

# In[154]:


# P2.1a
bearing = pd.read_csv("./bearing.csv", header=0, index_col=0) # 40 samples raw data
wearlevel = pd.read_csv("./wearlevel.csv", header=None, names=["wear"]) # wear level
bearing # 40 time series signal


# In[256]:


B = np.array(bearing) # transform dataframe bearing to an array

# Create centered data
B_center = np.zeros([len(B), len(B[0])]) # Create a array with same size as B
for i in range(len(B)):  # Loop over samples
    B_center[i] = B[i] - np.mean(B[i])  # Subtract the mean of each row from its values


    
B_center_df = pd.DataFrame(B_center, columns=bearing.columns)
B_center_df




# In[307]:


# PCA



from sklearn import decomposition
L = 30 # Plot L PCs in scree plot
pca = decomposition.PCA(n_components=L)
alpha = 0.1 # alpha value, used for explained variance ratio criteria


# Your code here to:
#   Fit in the data
#   Project the data to new plane
#   Get explaind variance ratio
#   Get the sum of explained variace ratio

pca.fit(B_center) # Fit in the data
PC = pca.transform(B_center) # Project the data to new plane
var_exp = pca.explained_variance_ratio_ # Get explaind variance ratio
cum_var_exp = np.cumsum(var_exp) # Get the sum of explained variace ratio

# Scree plot
plt.bar(range(L), var_exp, alpha=0.3, align='center', label='individual explained variance')
#plt.step(range(L), cum_var_exp, where='mid', label='cumulative explained variance')
plt.plot(np.arange(L), cum_var_exp, ".-",label='cumulative explained variance')
plt.plot([0, L], [0.9, 0.9], label="90%")
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.grid()
plt.legend(loc='best')
# plt.savefig("Bearing_PCA_center.png")


# In[308]:


index = np.argmax(cum_var_exp >=0.9)+1
print ( "Components to keep: ",index)


# In[276]:


# 3D scatter plot
col = ["blue", "red", "orange", "green"] # You need four color for four level

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot each data point
for i in range(len(PC)):
    w = wearlevel['wear'][i]
    ax.scatter(PC[i, 0], PC[i, 1], PC[i, 2] ,c=col[w])

custom_legend = [plt.Line2D([], [], marker='o', color=col[i], label=labels[i]) for i in range(4)]

# Label the axes
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.legend(handles=custom_legend)


plt.show()
PC[1]


# In[299]:


# Problem 3.2 FFT
F = np.zeros([len(B), 2])
threshold = 0.00
fft_intensity = np.zeros([40, 2048])

for i in range(len(F)):
    signal = B_center[i]
    Fs = 1280 # Sampling rate
    N = len(signal)
    T =  N/Fs # Duration
    yf = np.fft.fft(signal)
    yif = np.fft.ifft(yf) # Inverse fft (For demo purpose)
    yf = yf/N # Normalize intensity
    yf = yf[:N//2] # One side is enough since the fft is symmetric
    xf = np.fft.fftfreq(N, 1/Fs)
    xf = xf[:N//2]
    
    # xf is the frequency, abs(yf) is the intensity
    peak_freq_intensity = np.max(abs(yf))
    peak_freq = xf[np.where(abs(yf) == peak_freq_intensity)]

    
    F[i, 0] = peak_freq_intensity # finds the peak intensity from the FFT for each sample 
    F[i, 1] = peak_freq # finds the coressponding frequency of the peak intensity from above
    
    
    
    fft_intensity[i]= abs(yf)
  

 
print(F)
fft_intensity


# In[ ]:


fft_intensity = np.zeros([40, 2048])


# In[278]:


# P3.3 2D feature plot
col = ["blue", "red", "orange", "green"]
labels = ["Wear Level 0", "Wear Level 1", "Wear Level 2", "Wear Level 3"]

# Initialize an empty figure
plt.figure()

for i in range(len(F)):
    # Extract feature values from F
    feature1 = F[i, 0]  # Feature 1 (peak intensity)
    feature2 = F[i, 1]  # Feature 2 (peak frequency)

    # Extract the wear level for the current sample from the wearlevel DataFrame
    wear_level = wearlevel["wear"][i]

    # Plot the feature values with a color based on wear level
    plt.scatter(feature1, feature2, c=col[wear_level], label=labels[wear_level], alpha=0.7)

custom_legend = [plt.Line2D([], [], marker='o', color=col[i], label=labels[i]) for i in range(4)]

# Add labels and a legend
plt.xlabel("Peak Intensity")
plt.ylabel("Peak Frequency")
plt.legend(handles=custom_legend)

# Show the plot
plt.show()


# In[279]:


# P3.4
# Create a feature pool. Include wear level, 3 PCs, 2 frequency-domain feature
Feature = pd.DataFrame(columns=["wear", "P1", "P2", "P3", "F1", "F2"])
Feature["wear"] = wearlevel["wear"]
Feature[["P1", "P2", "P3"]] = PC[:, 0:3]
Feature[["F1", "F2"]] = F

# Two extra "row" to store Fisher ratio
Feature.loc["Fisher 1"] = ""
Feature.loc["Fisher 2"] = ""

# Seperate data to four dataframe for easy access
F0 = Feature[Feature.wear == 0]
F1 = Feature[Feature.wear == 1]
F2 = Feature[Feature.wear == 2]
F3 = Feature[Feature.wear == 3]

# Calculate Fisher ratio for each "feature"
for feature in Feature.columns[1:]:
    m0 = np.mean(F0[feature])
    m1 = np.mean(F1[feature])
    m2 = np.mean(F2[feature])
    m3 = np.mean(F3[feature])
    s0 = np.std(F0[feature])
    s1 = np.std(F1[feature])
    s2 = np.std(F2[feature])
    s3 = np.std(F3[feature])
    
    # Calculate Fisher's ratio 1 (level 0 - level 3)
    Fisher_ratio_1 = ((m0 - m3) ** 2) / (s0 ** 2 + s3 ** 2)
    
    # Calculate Fisher's ratio 2 for "level 1 - level 2"
    Fisher_ratio_2 = ((m1 - m2) ** 2) / (s1 ** 2 + s2 ** 2)
    
    # Store the Fisher ratios in the respective rows
    Feature.loc["Fisher 1", feature] = Fisher_ratio_1
    Feature.loc["Fisher 2", feature] = Fisher_ratio_2
    

# Extra "row" for mixed Fisher ratio
Feature.loc["Fisher mix"] = Feature.loc["Fisher 1"] + Feature.loc["Fisher 2"]
# Take a look at your current dataframe
Feature


# In[280]:


# 3D scatter plot
col = ["blue", "red", "orange", "green"] # You need four color for four level

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot each data point
for i in range(len(PC)):
    w = wearlevel['wear'][i]
    ax.scatter(F[i, 0], F[i, 1], PC[i, 0] ,c=col[w])

custom_legend = [plt.Line2D([], [], marker='o', color=col[i], label=labels[i]) for i in range(4)]

# Label the axes
ax.set_xlabel("Peak Intensity")
ax.set_ylabel("Peak Frequenxy")
ax.set_zlabel("PC1")

plt.legend(handles=custom_legend)


plt.show()
PC[1]
PC


# In[283]:


# FFT visualize
signal = B_center[5] # signal is a dataframe
Fs = 1280 # Sampling rate
N = len(signal)
T =  N/Fs # Duration
yf = np.fft.fft(signal)
yif = np.fft.ifft(yf) # Inverse fft (For demo purpose)
yf = yf/N # Normalize intensity
yf = yf[:N//2] # One side is enough since the fft is symmetric

xf = np.fft.fftfreq(N, 1/Fs)
xf = xf[:N//2]

plt.figure(figsize=(16, 8))
plt.subplot(211)
plt.plot(np.linspace(0, T, N), signal, "b")
plt.grid()
plt.xlabel("Time (ms)")
plt.ylabel("Accelerometer  (V)")
plt.subplot(212)
plt.plot(xf, abs(yf), "c-")
plt.grid()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Intensity')


# In[305]:


# PCA



from sklearn import decomposition
L = 30 # Plot L PCs in scree plot
pca = decomposition.PCA(n_components=L)
alpha = 0.1 # alpha value, used for explained variance ratio criteria

fft_row_means = np.mean(fft_intensity, axis=1)
fft_center = fft_intensity - fft_row_means[:,np.newaxis]


# Your code here to:
#   Fit in the data
#   Project the data to new plane
#   Get explaind variance ratio
#   Get the sum of explained variace ratio

pca.fit(fft_center) # Fit in the data
PC = pca.transform(fft_center) # Project the data to new plane
var_exp = pca.explained_variance_ratio_ # Get explaind variance ratio
cum_var_exp = np.cumsum(var_exp) # Get the sum of explained variace ratio

# Scree plot
plt.bar(range(L), var_exp, alpha=0.3, align='center', label='individual explained variance')
#plt.step(range(L), cum_var_exp, where='mid', label='cumulative explained variance')
plt.plot(np.arange(L), cum_var_exp, ".-",label='cumulative explained variance')
plt.plot([0, L], [0.9, 0.9], label="90%")
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.grid()
plt.legend(loc='best')
# plt.savefig("Bearing_PCA_center.png")

index = np.argmax(cum_var_exp >=0.9)+1
print ( "Components to keep: ",index)

fft_center.shape


# In[310]:


index = np.argmax(cum_var_exp >=0.9)+1
print ( "Components to keep: ",index)

