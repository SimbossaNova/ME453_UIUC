#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition
import pandas as pd

# Plot parameter setting 
plt.style.use('default')
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['figure.figsize'] = 8, 6
plt.rcParams['figure.dpi'] = 100


# The following code is almost fully functional. There are several lines ask you to modify the code. Please take a look at the comment and follow it.

# In[4]:


# Load data
training_set = pd.read_csv("./HW6_feature_training.csv", header=0, index_col=0)
testing_set = pd.read_csv("./HW6_feature_testing.csv", header=0, index_col=0)

testing_set
training_set


# In[5]:


Feature = pd.concat([training_set, testing_set])
Feature.loc["Fisher 1"] = ""
Feature.loc["Fisher 2"] = ""

F0 = Feature[Feature.wear == 0]
F1 = Feature[Feature.wear == 1]
F2 = Feature[Feature.wear == 2]
F3 = Feature[Feature.wear == 3]

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
    
    
    Feature.loc["Fisher 1", feature] = Fisher_ratio_1
    Feature.loc["Fisher 2", feature] =  Fisher_ratio_2

Feature.loc["Fisher compound"] = Feature.loc["Fisher 1"] + Feature.loc["Fisher 2"] 

Feature


# In[6]:


# Fisher plot
barwidth = 0.4
x1 = np.arange(0, 5)
x2 = [x + barwidth for x in x1]
plt.bar(x1, Feature.loc["Fisher compound"][1:], width=barwidth, label="Fisher compound")
#plt.bar(x2, Feature.loc["Fisher 2"][1:], width=barwidth, label="Fisher 2")
plt.xticks(x1)

# Your code to add legend, label and anyother useful information
plt.legend()
plt.title("Feature Index")
plt.ylabel("Fisher's Ratio")


# In[7]:


#1.3 Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

selected = ["0", "1","2","3","4"] # Modified the code to make sure it is the feature index you choose

x_train = training_set[selected]
y_train = training_set["wear"]
x_test = testing_set[selected]
y_test = testing_set["wear"]

clf_lda = LinearDiscriminantAnalysis()
clf_qda = QuadraticDiscriminantAnalysis()
clf_svm = SVC(kernel="rbf") # Modified the kernel to see the difference
clf_knn = KNeighborsClassifier(n_neighbors=5) # Modified the K to see the difference

clf_lda.fit(x_train, y_train)
clf_qda.fit(x_train, y_train)
clf_svm.fit(x_train, y_train)
clf_knn.fit(x_train, y_train)

# Change the classifier to show different result
plot_confusion_matrix(clf_lda, x_test, y_test)
print("LDA Classification Report")
accuracy_score(clf_lda.predict(x_test), y_test)


# In[8]:


#1.3 Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

selected = ["0", "1","2","3","4"] # Modified the code to make sure it is the feature index you choose

x_train = training_set[selected]
y_train = training_set["wear"]
x_test = testing_set[selected]
y_test = testing_set["wear"]

clf_lda = LinearDiscriminantAnalysis()
clf_qda = QuadraticDiscriminantAnalysis()
clf_svm = SVC(kernel="rbf") # Modified the kernel to see the difference
clf_knn = KNeighborsClassifier(n_neighbors=5) # Modified the K to see the difference

clf_lda.fit(x_train, y_train)
clf_qda.fit(x_train, y_train)
clf_svm.fit(x_train, y_train)
clf_knn.fit(x_train, y_train)

# Change the classifier to show different result
plot_confusion_matrix(clf_qda, x_test, y_test)
print("QDA Classification Report")

accuracy_score(clf_qda.predict(x_test), y_test)


# In[9]:


#1.4 Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

selected = ["2","3"] # Modified the code to make sure it is the feature index you choose

x_train = training_set[selected]
y_train = training_set["wear"]
x_test = testing_set[selected]
y_test = testing_set["wear"]

clf_lda = LinearDiscriminantAnalysis()
clf_qda = QuadraticDiscriminantAnalysis()
clf_svm = SVC(kernel="rbf") # Modified the kernel to see the difference
clf_knn = KNeighborsClassifier(n_neighbors=5) # Modified the K to see the difference

clf_lda.fit(x_train, y_train)
clf_qda.fit(x_train, y_train)
clf_svm.fit(x_train, y_train)
clf_knn.fit(x_train, y_train)

# Change the classifier to show different result
plot_confusion_matrix(clf_qda, x_test, y_test)
print("QDA Classification Report for Two Highest Fisher Rations ")

accuracy_score(clf_qda.predict(x_test), y_test)


# In[10]:


#1.4 Classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

selected = ["2","3"] # Modified the code to make sure it is the feature index you choose

x_train = training_set[selected]
y_train = training_set["wear"]
x_test = testing_set[selected]
y_test = testing_set["wear"]

clf_lda = LinearDiscriminantAnalysis()
clf_qda = QuadraticDiscriminantAnalysis()
clf_svm = SVC(kernel="rbf") # Modified the kernel to see the difference
clf_knn = KNeighborsClassifier(n_neighbors=5) # Modified the K to see the difference

clf_lda.fit(x_train, y_train)
clf_qda.fit(x_train, y_train)
clf_svm.fit(x_train, y_train)
clf_knn.fit(x_train, y_train)

# Change the classifier to show different result
plot_confusion_matrix(clf_lda, x_test, y_test)
print("LDA Classification Report for Two Highest Fisher Rations ")

accuracy_score(clf_lda.predict(x_test), y_test)


# In[12]:


#################################################
#################################################


#1.5

#        Based on the answers from 1.3 and 1.4, adding more features does not neccessarily improve  
#        the classification performance
















# In[22]:


# 1.6



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
sorted_features = Feature.loc["Fisher compound"][1:].sort_values(ascending=False)
i=0
train_error_lda = []
test_error_lda = []
train_error_qda = []
test_error_qda = []
# Selecting top 5 features based on Fisher's ratios
selected_features = sorted_features.index[:5]


x_train = training_set[selected_features]
y_train = training_set["wear"]
x_test = testing_set[selected_features]
y_test = testing_set["wear"]

for n_features in range(1, 6):  # n = 1 to 5
    top_n_features = selected_features[:n_features]  # Select top n features
    x_train_n = x_train[top_n_features]
    x_test_n = x_test[top_n_features]

    # Train LDA and QDA models
    lda = LinearDiscriminantAnalysis()
    qda = QuadraticDiscriminantAnalysis()

    lda.fit(x_train_n, y_train)
    qda.fit(x_train_n, y_train)

    # Calculate training and test classification errors
    train_error_lda.append(1 - lda.score(x_train_n, y_train))
    test_error_lda.append(1 - lda.score(x_test_n, y_test))

    train_error_qda.append(1 - qda.score(x_train_n, y_train))
    test_error_qda.append(1 - qda.score(x_test_n, y_test))
    
    # Print classification errors
    print(f"Number of Features: {n_features}")
    print("Features nos considered", top_n_features)
    print(f"Training Error (LDA): {train_error_lda[i]:.4f}")
    print(f"Test Error (LDA): {test_error_lda[i]:.4f}")
    print(f"Training Error (QDA): {train_error_qda[i]:.4f}")
    print(f"Test Error (QDA): {test_error_qda[i]:.4f}")
    print("----------------------------------------")
    i+=1


# Generating a list of numbers from 1 to 5 (number of features)
no_of_features = list(range(1, 6))

# Plotting the array against the number of features for LDA
plt.figure(figsize=(12, 6))  # Adjust figure size as needed
plt.plot(no_of_features, test_error_lda, marker='o', linestyle='-', color='r', label='Test Error (LDA)')
plt.plot(no_of_features, train_error_lda, marker='o', linestyle='-', color='b', label='Train Error (LDA)')
plt.xticks(no_of_features)
plt.xlabel('Number of Features')
plt.ylabel('Prediction Error')
plt.title('Prediction Error vs Number of Features for LDA')
plt.grid(True)
plt.show()


# Plotting the array against the number of features for LDA
plt.figure(figsize=(12, 6))  # Adjust figure size as needed
plt.plot(no_of_features, test_error_qda, marker='o', linestyle='-', color='r', label='Test Error (QDA)')
plt.plot(no_of_features, train_error_qda, marker='o', linestyle='-', color='b', label='Train Error (QDA)')
plt.xticks(no_of_features)
plt.xlabel('Number of Features')
plt.ylabel('Prediction Error')
plt.title('Prediction Error vs Number of Features for QDA')
plt.grid(True)
plt.show()


# In[14]:


from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

selected = ["3","2"] # Modified the code to make sure it is the feature index you choose


x_train = training_set[selected]
y_train = training_set["wear"]
x_test = testing_set[selected]
y_test = testing_set["wear"]

kernels = ["linear", "poly", "rbf"]

for kernel in kernels:
    if kernel == "poly":
        clf = SVC(kernel=kernel)
    else:
        clf = SVC(kernel=kernel)

    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Kernel: {kernel}")
    print(f"Accuracy: {accuracy:.2f}")
    
    plot_confusion_matrix(clf, x_test, y_test)
    plt.title(f"Confusion Matrix for {kernel} Kernel")
    plt.show()


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

selected = ["2","3"] # Modified the code to make sure it is the feature index you choose


x_train = training_set[selected]
y_train = training_set["wear"]
x_test = testing_set[selected]
y_test = testing_set["wear"]

for k in [1, 5]:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"KNN with K = {k}")
    print(f"Accuracy: {accuracy:.2f}")
    
    plot_confusion_matrix(knn, x_test, y_test)
    plt.title(f"Confusion Matrix for KNN with K={k}")
    plt.show()


# In[16]:


###############





# 1.7

# The QDA classication has the best performance of 0.875 accuracy compared to the other methods 
# that have accuracies of 0.833 and lower 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




