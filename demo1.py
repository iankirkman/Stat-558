'''
Run a demo of my Elastic Net implementation with real-world data
'''

import numpy as np
import pandas as pd
import sklearn.preprocessing
import matplotlib.pyplot as plt
import elastic_net.py

# Load and standardize Hitters data
hitters = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/Hitters.csv", sep=',',header=0)
hitters = hitters.dropna()

# Create our X matrix with the predictors and Y vector with the response
X = hitters.drop('Salary',axis=1)
Y = hitters.Salary

# Encode the variables League, Division, and Newleague
X = pd.get_dummies(X, drop_first=True)

# Standardize the data
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
Y = np.array(Y-np.mean(Y))/np.std(Y)

# In general, you will just want to use the elastic_net function
# This returns a tuple of the lambda used in the model, and the beta vector of coefficients

# With defaults:
elastic_net(X,Y)

# Specifying a lamda grid for CV search:
opt_lam,beta = elastic_net(X,Y,lam=[10**i for i in range(-3,3)])

# To see the convergence of the coordinate descent, we can call other package functions and plot

# Let's look at the objective function versus iterations:
ran = randcoorddescent(np.zeros(X.shape[1]),1000,X,Y,opt_lam,.9)
ran_objs = [computeobj(ran[:,i],X,Y,opt_lam,.9) for i in range(1000)]
# Plot them
plt.plot(range(0,1000),ran_objs)
plt.show()
