'''
Compare results to scikit-learn's ElasticNetCV function.

Note that the lambda/alpha parameters are different in sklearn's objective function.
Scikit alpha_ = Our lambda/2
Scikit l1_ratio_ = Our alpha
'''

import numpy as np
import matplotlib.pyplot as plt
import elastic_net

from sklearn.linear_model import ElasticNetCV

# Create standardized, simulated data
X = np.diag(np.random.uniform(5,10,100))
X = (X-np.mean(X))/np.std(X)
Y = np.random.normal(0,1,100)

# Fit the scikit-learn model
sk = ElasticNetCV()
sk.fit(X,Y)

# Print the coefficients
print(sk.coef_)

# Use scikit's grid search results to set our parameters
print(elastic_net(X,Y,lam=2*sk.alpha_,a=sk.l1_ratio_)[1])

# We see that the resuls are similar, but not perfectly matched

# Now let's run our grid search on lambda to see if we find a similar optimal parameter
print('Sklearn optimal param: %f'%(2*sk.alpha_))

opt_lam = elastic_net(X,Y,lam=[.01*i for i in range(1,50)],a=sk.l1_ratio_)[0]
print('Our optimal param: %f'%opt_lam)
