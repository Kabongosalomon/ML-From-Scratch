# =============================================================================
# Dimby 
# =============================================================================


import numpy as np

"""
# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================
"""

class logistic_regression():
    def __init__(self, lr = 0.1, tolerence = 1e-5, iteration = 500):
        self.lr = lr
        self.tolerence = tolerence
        self.num_iter = iteration
   
    def dot_prod(self, x, theta):
        return np.dot(x,theta)
    
    def Sigmoid(self, x):
        return 1.0/(1+np.exp(-x))
    
    def loss(self, y, h):
        l=(y*np.log(h) + (1-y)*np.log(1-h)).mean()
        return l
    
    def Gradient(self, h, x, y):
        g = x*((y -h).reshape(-1,1))
        return g
    
    def fit(self, x, y):
        np.concatenate((x, np.ones((len(x),1))), axis = 1)
        self.theta = np.random.randn(x.shape[1])
        alpha_ = self.lr
        n = self.num_iter
        for i in range(n):
            previous_theta = self.theta
            X = self.dot_prod(x,self.theta)
            h = self.Sigmoid(X)
            gradient = self.Gradient(h,x,y)
            previous_theta, self.theta = self.theta, self.theta +\
            alpha_*(gradient.sum(axis = 0))
            if np.linalg.norm(self.theta - previous_theta) < self.tolerence:
                print('Horray!!')
                break
            l = self.loss(y,h)
            if i%100 == 0:
                print('lOSS {}: {}'.format(i,l))
            
    def predict(self, x):
        r = self.Sigmoid(self.dot_prod(x,self.theta))
        return np.round(r)


"""
# =============================================================================
# LINEAR REGRESSION
# =============================================================================
"""


class linear_regression():
    def __init__(self, lr = 0.1, tolerence = 1e-5, iteration = 500):
        self.lr = lr
        self.tolerence = tolerence
        self.num_iter = iteration
   
    def dot_prod(self, x, theta):
        return np.dot(x,theta)
    
    def loss(self, y, h):
        l=((y-h)**2).mean()
        return l
    
    def Gradient(self, h, x, y):
        g = x*((y -h).reshape(len(y),1))
        return g
    
    def fit(self, x, y):
        np.concatenate((x, np.ones((len(x),1))), axis = 1)
        self.theta = np.random.randn(x.shape[1])
        alpha_ = self.lr
        n = self.num_iter
        for i in range(n):
            previous_theta = self.theta
            h = self.dot_prod(x,self.theta)
            gradient = self.Gradient(h,x,y)
            previous_theta, self.theta = self.theta, self.theta + alpha_*(gradient.sum(axis = 0))
            if np.linalg.norm(self.theta - previous_theta) < self.tolerence:
                print('Horray!!')
                break
            l = self.loss(y,h)
            print(l)
    def predict(self, x):
        p = self.dot_prod(x,self.theta)
        return p



