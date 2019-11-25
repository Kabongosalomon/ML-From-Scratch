import numpy as np
from numpy.linalg import inv
import scipy.stats


class GaussianDiscriminantAnalysis:
    def __init__(self):
        self.phi = []
        self.sigma = 0
        self.u = []
        self.labels = 0
        self.n = 0
    
    def calculate_phi(self, y):
        for i in range(len(self.labels)):
            self.phi.append((y==self.labels[i]).mean())
    
    def calculate_sigma(self, X, y):
        s = 0
        for i in range(len(y)):
            val = np.dot((X[i].reshape(-1, 1)-self.u[y[i]]),(X[i].reshape(-1, 1)-self.u[y[i]]).T) 
            s += val

        return s/len(y)

    def calculate_u(self, X, y):
        for i in range(len(self.labels)):
            self.u.append((np.dot((y==self.labels[i]), X)/len(y[y==self.labels[i]])).reshape(X.shape[1], -1))
    
    def calculate_mu_1(self, X, y):
        return np.dot(1*(y==self.labels[1]), X)/len(y[y==self.labels[1]])/len(y)
    
    def fit(self, X, y):
        self.n = len(y)
        self.labels = np.unique(y)
        
        self.calculate_u(X, y)
        
        self.calculate_phi(y)
        self.sigma = self.calculate_sigma(X, y)
        
    
    def calculate_px_py(self, x):
        pi = np.pi
        dim = len(x)
        pxpy = []
        for i in range(len(self.labels)):
            pxpy.append((1/(2*pi)**(dim/2)*np.sqrt(np.linalg.det(self.sigma)))*\
                        np.exp(np.dot(np.dot((-0.5*(x-self.u[i]).T), inv(self.sigma)), (x-self.u[i]))))
        
        return pxpy

    def calculate_py(self, y):
        for i in range(n):
            self.phi.append((y==self.labels[i]).mean())
            
        return self.phi if y==labels[1] else (1-self.phi)
        
    
    def predict(self, X):
        total = []
        for i_ in range(self.n):
            
            result = []
            for i in range(len(self.labels)):
                result.append(self.calculate_px_py(X[i_].reshape(-1, 1))[i]*self.phi[i])
            total.append(np.argmax(result))
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