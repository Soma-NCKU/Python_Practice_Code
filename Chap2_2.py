# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:54:44 2020

Reference Textbook : Python Machine Learning 
                Sebastian & vahid

"""

import numpy as np
import matplotlib.pyplot as plt

class AdalineGD():
    """ ADApative LInear NEuron classifier.
    
    Parameters
    -------------
    eta : float
        Learning rate (between 0.0 and 1.0)
    n_iter : int
        Passes over the traning dataset
    random_state : int
        Random number generator seed for random weight initialization.
        
    Attributes
    -------------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list 
        Sum-of-squares cost function value in each epoch.
    
    """
    
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self, X, y):
        """ Fit training data.
        
        Parameters
        -------------
        X : {array-like}, shape = [n_smaples, n_features]
            Training vectors, where n_samples is the number of samples and 
            n_features is the number of features.
        y : array-like, shape = [n_smaples]
            Target values.
            
        Returns
        -----------
        self : object
        
        """
        
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0, scale = 0.01, size = 1 + X.shape[1])
        self.cost_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, X):
        """Compute linear activation"""
        return X
    
    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
    
# ============================================================================

# Data

import pandas as pd 

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                 'machine-learning-databases/iris/iris.data',
                 header = None)
df.tail()

## tail() function is used to get the last n rows. This function returns last n rows from the object based on position.

print(type(df))

df

# ============================================================================

# Data Visualization


# select setosa and versicolor
y = df.iloc[0:100, 4].values     #iloc[row, col].values - in the form of array; without ".values" - it will be list
y = np.where(y == 'Iris-setosa', -1, 1)

## loc in Pandas is label-based, which means that we have yo specify the name of the rows and cols that we need to filter out whereas 
## iloc in Pnadas is integer index- based, we have to specify rows and cols by their integer index.

# extract sepal length and petal length

X = df.iloc[0:100,[0,2]].values

# plot data

plt.scatter(X[:50, 0], X[:50, 1],
            color = 'red', marker = 'o', label = 'setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color = 'blue', marker = 'x', label = 'versicolor')

plt.xlabel(' sepal length [cm] ')
plt.ylabel(' petal length [cm] ')

plt.legend(loc = 'upper left')
plt.show()

# ============================================================================


fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))

ada1 = AdalineGD(n_iter = 10, eta = 0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker = 'o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline-Learning rate 0.01')

ada2 = AdalineGD(n_iter = 10, eta = 0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_) + 1), ada2.cost_, marker = 'o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
ax[1].set_title('Adaline-Learning rate 0.0001')
plt.show()

# ============================================================================

# Imporving the gradient descent through feature scaling

# Graident descent is one of the many algorithm that nenefits from feature scaling. Standardization - is a feature scaling method
# which gives data the property of a standard normal distribution, which helps gradient descent learning to converge more quickly.

# Standardization shifts the mean of each feature so that it is centered at zero and each feature has a standard deviation of 1. 
# Reason why standardization helps with GD learning is that the optimizer has to go through fewer steps to find a good or optimal 
# solution. 

# Standarization can easily be achieved using the built-in NumPy methods "mean" and "std"

X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

ada = AdalineGD(n_iter = 15, eta = 0.01)
ada.fit(X_std, y)


# ============================================================================

# Plot the convergence for the algorithm

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution = 0.02):
    
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
## determine the minimum and maximum values for the two features 
## and use those features vectors to create a pair of grid arrays xx1 and xx2

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
## Since we trained preceptron classifier on two feature dimensions, we 
## need to flatten the grid arrays and create a matrix that has the same 
## number of columns as the Iris training subset so that we can use the 
## predict method to predict the class labels Z of the corresponding grid points.

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')
        

# ===========================================================================

plot_decision_regions(X_std, y, classifier = ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardization]')
plt.ylabel('petal length [standardization]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
plt.show()

            
        
