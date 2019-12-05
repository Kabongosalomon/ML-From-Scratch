import numpy as np
from numpy.linalg import inv
import scipy.stats
import ipdb


class GaussianDiscriminantAnalysis:
    def __init__(self, epsilon=1e-9):
        self.epsilon = epsilon
        self.phi = []
        self.sigma = 0
        self.u = []
        self.labels = 0
        self.n = 0
        self.n_label = 0
        self.map_features = {}
    
    def calculate_phi(self, y):
        for i in range(self.n_label):
            self.phi.append((y==self.labels[i]).mean())
    
    def calculate_sigma(self, X, y):
        s = 0
        for i in range(self.n):
            val = np.dot((X[i].reshape(-1, 1)-self.u[self.map_features[y[i]]]),\
                         (X[i].reshape(-1, 1)-self.u[self.map_features[y[i]]]).T) 
            s += val

        return s/len(y)

    def calculate_u(self, X, y):
        for i in range(self.n_label):
            self.u.append((np.dot((y==self.labels[i]), X)/len(y[y==self.labels[i]])).reshape(X.shape[1], -1))
    
    
    def fit(self, X, y):
        self.n = len(y) # Number of examples
        
        self.labels = np.unique(y)
        # To handle the case where the y is a float
        self.labels = self.labels.astype('i')        
        self.n_label = len(self.labels)
        self.map_features = {v:k for k,v in enumerate(self.labels)} # trick to deal with non zeros based target
#         self.map_features{self.labels}=range(self.n)
        
        self.calculate_u(X, y)
        self.calculate_phi(y)
        self.sigma = self.calculate_sigma(X, y)
        
    
    def calculate_px_py(self, x):
        pi = np.pi
        dim = len(x)
        pxpy = []
        for i in range(self.n_label):
            # We are adding self.epsilon to avoid non inverstible matrix
#             ipdb.set_trace()
            pxpy.append((1/(2*pi)**(dim/2)*np.sqrt(np.linalg.det(self.sigma)))*\
                        np.exp(np.dot(np.dot((-0.5*(x-self.u[i]).T), np.linalg.pinv(self.sigma)), (x-self.u[i]))))
        
        return pxpy

    def calculate_py(self, y):
        for i in range(n):
            self.phi.append((y==self.labels[i]).mean())
            
        return self.phi if y==labels[1] else (1-self.phi)
        
    
    def predict(self, X):
        total = []
        for i_ in range(X.shape[0]):
            
            result = []
            for i in range(self.n_label):
                result.append(self.calculate_px_py(X[i_].reshape(-1, 1))[i]*self.phi[i])
            total.append(self.labels[np.argmax(result)]) # nice trick to solve non zeros based issue 
        return np.array(total)
    
    def predict_prob(self, X):
        total = []
        for i_ in range(self.n):
            
            result = []
            for i in range(len(self.labels)):
                result.append(self.calculate_px_py(X[i_].reshape(-1, 1))[i]*self.phi[i])
            total.append(result)
        return np.array(total)
    
    def accuracy(self, y, y_hat):
        count=0
        for i in range(len(y)):
            if y[i]==y_hat[i]:
                count+=1
        return count/len(y)