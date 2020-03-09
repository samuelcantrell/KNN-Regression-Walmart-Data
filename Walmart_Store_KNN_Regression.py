# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 13:43:13 2020

@author: scant
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class KNN_Classifier():
    def __init__(self, X, y):  # Fit function
        self.X = X
        self.y = y

    def predict(self, X, k, epsilon):  # Calculate probability on new input
        N = len(X)
        y_hat = np.zeros(N)
        for i in range(N):
            dist2 = np.sum((self.X-X[i])**2, axis=1)
            idxt = np.argsort(dist2)[:k]
            gamma_k = (np.sqrt(dist2[idxt]) + epsilon)**-1
            y_hat[i] = np.bincount(self.y[idxt], weights=gamma_k).argmax()

        return y_hat  # return y_hat, i.e. argmax(P_hat)


class KNN_Regressor():
    def __init__(self, X, y):  # Fit function
        self.X = X
        self.y = y

    def predict(self, X, K, epsilon):
        N = len(X)
        y_hat = np.zeros(N)

        for i in range(N):
            dist2 = np.sum((self.X - X[i])**2, axis=1)*epsilon  # Normalize a bit
            idxt = np.argsort(dist2)[:K]
            gamma_k = np.exp(-dist2[idxt])/(np.exp(-dist2[idxt]).sum() + epsilon)
            # print(np.exp(-dist2[idxt]))
            y_hat[i] = gamma_k.dot(self.y[idxt])

        return y_hat


class GeneralizedRegression():
    def __init__(self, X, Y):
        X_temp = np.insert(X, 0, 1, axis=1)
        self.w = np.linalg.lstsq(np.dot(X_temp.T, X), np.dot(X_temp.T, Y))[0]

    def predict(self, X):
        return np.dot(X, self.w)

def accuracy(y, y_hat):
    return np.mean(y == y_hat)


data = pd.read_csv("cleaned_walmart_store_data_train.csv")
# print(data.head())
X = data.to_numpy()

# Build list of KNN_Regressor objects for each store.
y_ws = X[:, 14:15]  # Define Weekly_Sales for Regression
X = X[:, :-1]  # Chop off last column of X now that we've grabbed the labels.
Reg_knn = []
Reg_sl = []

for i in range(1, len(np.unique(X[:, 1])) + 1):  # For each unique store.
    Reg_knn.append(KNN_Regressor(X[X[:, 1] == i], y_ws[X[:, 1] == i]))  # Make a unique regressor
    Reg_sl.append(GeneralizedRegression(X[X[:, 1] == i], y_ws[X[:, 1] == i]))  # Make a unique regressor

# Then build Store Select object from KNN_Classifier to feed to the regressor.
y_st = X[:, 1].astype(int, casting="unsafe")  # Define Store for Classifier.
X = np.delete(X, 1, axis=1)  # Get rid of Store labels from main set.

St_Sel = KNN_Classifier(X, y_st)

# Now evaluate the test sets. ----------
k1 = 99
k2 = 5
eps = 1e-10

data = pd.read_csv("cleaned_walmart_store_data_testyear3.csv")
# print(data.head())
X = data.to_numpy()
y_ws = X[:, -1]  # Define Weekly_Sales for Regression
X = X[:, :-1]  # Chop off last column of X now that we've grabbed the labels.
y_st = X[:, 1]  # Define Store for Classifier.
X = np.delete(X, 1, axis=1)  # Get rid of Store labels from main set.

s_temp = St_Sel.predict(X, k1, eps).astype(int, casting="unsafe")

print(accuracy(y_st, s_temp))  # Test Classification
plt.figure(figsize=(12, 8))
y_true = pd.Series(y_st, name="Actual Store")
y_pred = pd.Series(s_temp, name="Predicted Store")
sns.heatmap(pd.crosstab(y_true, y_pred), annot=True, fmt="d", linewidths=0.005)
plt.ylim(len(set(y_st)), 0)  # Fix limits, matplotlib bugged (ver. 3.11)

# Now evaluate new stores.
data = pd.read_csv("cleaned_walmart_store_data_testnewstores.csv")
# print(data.head())
X = data.to_numpy()
y_ws = X[:, -1]  # Define Weekly_Sales for Regression
X = X[:, :-1]  # Chop off last column of X now that we've grabbed the labels.
y_st = X[:, 1]  # Define Store for Classifier.
X = np.delete(X, 1, axis=1)  # Get rid of Store labels from main set.

s_temp = St_Sel.predict(X, k1, eps).astype(int, casting="unsafe")

# Begin Prediction
X = np.insert(X, 1, y_st, axis=1)  # Add predicted stores for prediction
R2 = []
r2 = 0

for i in np.unique(y_st):  # For each unique store predicted, evaluate
    X_temp = X[X[:, 1] == i]
    y_temp = y_ws[X[:, 1] == i]
    st = int(np.mean(s_temp[X[:, 1] == i]))

    y_hat = Reg_knn[st-1].predict(X_temp, k2, eps)
    R2.append(np.corrcoef(y_temp, y_hat)[0, 1]**2)
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(X_temp[:, 0], y_temp, s=5)
    plt.plot(X_temp[:, 0], y_hat, color='#000000')
    plt.xlabel('Week Number')
    plt.ylabel('Weekly Sales')
    """
# print(R2)
r2 = np.max(R2)
print("KNN R^2 Value: " + str(r2))

R2 = []
for i in np.unique(y_st):  # For each unique store predicted, evaluate
    X_temp = X[X[:, 1] == i]
    y_temp = y_ws[X[:, 1] == i]
    st = int(np.mean(s_temp[X[:, 1] == i]))

    y_hat = Reg_sl[st-1].predict(X_temp)
    y_hat = y_hat[:, 0]
    R2.append(np.corrcoef(y_temp, y_hat)[0, 1]**2)
    """
    plt.figure(figsize=(12, 8))
    plt.scatter(X_temp[:, 0], y_temp, s=5)
    plt.plot(X_temp[:, 0], y_hat, color='#000000')
    plt.xlabel('Week Number')
    plt.ylabel('Weekly Sales')
    """
# print(R2)
r2 = np.max(R2)
print("SLR R^2 Value: " + str(r2))
