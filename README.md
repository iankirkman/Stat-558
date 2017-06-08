# Stat-558 Elastic Net Implementation

Author: Ian Kirkman (ikirkman@uw.edu)

Date: 6/7/2017

This module implements the Elastic Net model with random coordinate decent, by iterating
over the partial minimization problem with respect to $\beta_j$:

$$min_{\beta_j} \frac{1}{n} ||Y-X\beta||^2_2 + \lambda(1-\alpha)||\beta||^2_2 + \alpha \lambda ||\beta||_1$$

The user has the option to use a default of 1, or tune the penalty lambda via k-folds
cross validation. (The default number of folds is 10). The alpha scalar of the elastic net
model has default value 0.9, and should fall within (0,1) if it is overwritten.

The stopping criteria is strictly based on the specified max iterations (default 1000). 

Note that input data should be standardized.
