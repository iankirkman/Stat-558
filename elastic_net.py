'''
This module implements the Elastic Net model with random coordinate decent, by iterating
over the partial minimization problem with respect to \beta_j:

min_{\beta_j} \frac{1}{n} ||Y-X\beta||^2_2 + \lambda(1-\alpha)||\beta||^2_2 + \alpha \lambda ||\beta||_1

The user has the option to use a default of 1, or tune the penalty lambda via k-folds
cross validation. (The default number of folds is 10). The alpha scalar of the elastic net
model has default value 0.9, and should fall within (0,1) if it is overwritten.

The stopping criteria is strictly based on the specified max iterations (default 1000).
'''

import numpy as np
import collections

def elastic_net(X,Y,max_iter=1000,lam=1,a=.9,k=10):
    '''
    Run the elastic net model
    
    Input:
        X: the matrix of predictors as a numpy array
        Y: the response vector as a numpy array
        max_iter: the number of iterations for the random coordinate descent (stopping criteria)
        lam: the penalty scalar for the elastic net model (can be a list for CV grid)
        a: the alpha scalar for the elastic net model
        k: the number of folds to use in CV lambda grid search
        
    Returns (lam,beta) tuple of:
        lam: the lambda used in model calculations (will be 1 if no grid is provided)
        beta: the regression coefficients calculated by the Elastic Net Model.
    '''
    if isinstance(lam, collections.Iterable):
        # Run Grid CV to find optimal lambda:
        lam = grid_lambdas(lam,X,Y,max_iter,a,k)[0]
    return lam,randcoorddescent(np.zeros(X.shape[1]), max_iter, X, Y, lam, a)[-1]
    
def soft_threshhold(c,alphalam):
    '''
    Helper function to allow sign operator in minimization problem.
    See partial_min() for context.
    '''
    if c < -alphalam:
        return c+alphalam
    elif c > alphalam:
        return c-alphalam
    else:
        return 0

def partial_min(j, beta, lam, a, X, Y):
    '''
    Compute the solution of the Elastic Net coordinate descent minimization at one iteration.

    Input:
        j: the coordinate index to optimize over
        beta: the vector of regression coefficients 
        lam: the lambda scalar in the elastic net function
        a: the alpha scalar in the elastic net function
        X: the matrix of predictors 
        Y: the response vector
    
    Returns: the new beta_j
    '''
    n, d = X.shape
    # Remove jth element from X and beta
    tempX, tempb = (np.copy(X),np.copy(beta))
    tempX[j] = False
    tempb[j] = False
    
    Rj = Y-tempX.dot(tempb.T)[j]
    zj = np.sum(X[:,j]**2)
    return soft_threshhold((2/n)*np.dot(X[:,j],Rj),a*lam)/((2/n)*zj+2*lam*(1-a))
    
def computeobj(beta,X,Y,lam,a):
    '''
    Computes the elastic net objective function.

    Input:
        beta: the coefficient vector to evaluate the objective function at
        X: the feature matrix (as a numpy array)
        Y: the numpy array of responses
        lam: the lambda scalar in the objective function
        a: the alpha scalar in the objective function

    Returns: the objective function value at beta
    '''
    return 1/len(Y)*np.dot((Y-X.dot(beta.T)),(Y-X.dot(beta.T)))+(1-a)*lam*np.sum(beta**2)+a*lam*np.linalg.norm(beta)
    
def pickcoord(d):
    '''
    Returns a uniform random sample from the set {1,...,d}

    Input: 
        d: the set parameter to draw the sample from
    '''
    return np.random.randint(low=1,high=d)
    
def randcoorddescent(b_init, max_iter, X, Y, lam, a):
    '''
    Randomized implementation of the coordinate descent algorithm for the elastic net function

    Inputs:
        b_init: initial point
        max_iter: maximum iterations (stopping criteria)
        X: the numpy array of features
        Y: the numpy array of responses
        lam: the lambda scalar in the elastic net function
        a: the alpha scalar in the elastic net function

    Returns: Arrary of points at each step in algorithm. Final array entry is the minimum calculated by the algorithm.
    '''
    # Create array to store beta at each iteration
    d = len(b_init)
    n = len(Y)
    bvals = np.empty((d,max_iter))
    
    # Initialize values
    j,old_j = (-1,-1)
    bvals[:,0] = b_init
    
    # Random Coordinate Descent
    for i in range(1,max_iter):
        # Choose a new index j randomly
        j = pickcoord(d)-1
        # Choose again if it's the same coordinate just descended
        while (j==old_j):
            j = pickcoord(d)-1
            
        # Update values
        bvals[:,i] = np.copy(bvals[:,i-1])
        bvals[j,i] = partial_min(j,bvals[:,i-1],lam,a,X,Y)
        
    return bvals
    
def kfolds_cv(X,Y,max_iter,lam,a,k):
    '''
    Implementation of k-folds cross validation for elastic net regression

    Inputs:
        X: the numpy array of features
        Y: the numpy array of responses
        max_iter: maximum iterations (stopping criteria)
        lam: the lambda scalar in the elastic net function
        a: the alpha scalar in the elastic net function
        k: number of folds

    Returns: average MSE over all folds
    '''
    # Shuffle data for random selection
    XY_cv = np.column_stack((X,Y))
    np.random.shuffle(XY_cv)

    # Set foldsize based on number of folds
    fsize = int(np.floor(XY_cv.shape[0]/k))
    
    # Create array to store MSE of each fold
    mses = np.empty(k)

    for fold in range(0,k):
        # For each fold, pull out (near-)equal size partition for validation set
        # If n is not divisible by the number of folds, the remainder will be included in the last group
        if fold == 0:
            X_test = XY_cv[:fsize,:19]
            X_train = XY_cv[fsize:,:19]
            Y_test = XY_cv[:fsize,19]
            Y_train = XY_cv[fsize:,19]
        elif fold == 9:
            X_test = XY_cv[fold*fsize:,:19]
            X_train = XY_cv[:fold*fsize,:19]
            Y_test = XY_cv[fold*fsize:,19]
            Y_train = XY_cv[:fold*fsize,19]
        else:
            X_test = XY_cv[fold*fsize:(fold+1)*fsize,:19]
            X_train = np.row_stack((XY_cv[:fold*fsize,:19],XY_cv[(fold+1)*fsize:,:19]))
            Y_test = XY_cv[fold*fsize:(fold+1)*fsize,19]
            Y_train = np.concatenate((XY_cv[:fold*fsize,19],XY_cv[(fold+1)*fsize:,19]))
            
        # Calculate MSE for fold
        fold_beta = randcoorddescent(np.zeros(X_train.shape[1]),max_iter,X_train,Y_train,lam,a)[:,-1]
        mses[fold] = np.sum((Y_test - X_test.dot(fold_beta))**2)/len(Y_test)
        
    return np.mean(mses)

def grid_lambdas(lambs,X,Y,max_iter,a,folds):
    '''
    Test grid of lambdas for smallest MSE
    Returns a (opt_lam,mse_grid) tuple of the optimal lambda and the grid of Mean Squared Errors

    Inputs:
        lambs: the lambdas to grid
    '''
    mses = np.empty(len(lambs))
    i = 0
    for lam in lambs:
        mses[i] = kfolds_cv(X,Y,max_iter,lam,a,folds)
        i += 1
    return (lambs[np.argmin(mses)],mses)
