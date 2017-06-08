'''
Run a demo of my Elastic Net implementation with simple simulated data.
'''

import numpy as np
import matplotlib.pyplot as plt
import elastic_net

# Create standardized, simulated data
X = np.diag(np.random.uniform(5,10,100))
X = (X-np.mean(X))/np.std(X)
Y = np.random.normal(0,1,100)

'''
In general, you will just want to use the elastic_net function
This returns a tuple of the lambda used in the model, and the beta vector of coefficients
'''

# With defaults:
elastic_net(X,Y)

# Specifying a lamda grid for CV search:
opt_lam,beta = elastic_net(X,Y,lam=[10**i for i in range(-3,3)])

'''
To see the convergence of the coordinate descent, we can call other package functions and plot
'''
# Let's look at the objective function versus iterations:
ran = randcoorddescent(np.zeros(X.shape[1]),1000,X,Y,opt_lam,.9)
ran_objs = [computeobj(ran[:,i],X,Y,opt_lam,.9) for i in range(1000)]
# Plot them
plt.plot(range(0,1000),ran_objs)
plt.show()

'''
We can also look at a plot of the MSE over log(lambda) for the grid:
'''
lambs = [10**i for i in range(-3,3)]
mses = grid_lambdas(lambs,X,Y,1000,.9,10)[1]
plt.plot(np.log(lambs),mses)
plt.show()
