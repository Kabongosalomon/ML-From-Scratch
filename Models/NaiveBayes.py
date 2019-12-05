import numpy as np
from numpy.linalg import inv
import scipy.stats

class MultinomialNB():
    def __init__(self):
        self.labels = None
        self.phi = []
        self.ph0_j = []
        self.ph1_j = []
    
    def calculate_phi_s(self, y):
        for i in range(len(self.labels)):
            self.phi.append((sum(y == self.labels[i])+1)/(len(y)+len(self.labels)))
            
    def compute_phi_x0y(self, X, y):
        for i in range(len(self.labels)):
            self.ph0_j.append((sum(X[y==self.labels[i]]==0)+1)/(sum(y == self.labels[i])+len(self.labels)))
            
    def compute_phi_x1y(self, X, y):
        for i in range(len(self.labels)):
            self.ph1_j.append((sum(X[y==self.labels[i]]==1)+1)/(sum(y == self.labels[i])+len(self.labels)))
            
            
    def prob_x_y(self, X):
        probab = []
        for k in range(len(self.labels)):
            real_prob = []
            for i in range(X.shape[0]):
                X_ = X[i, :]
                prod = 1
                for p in range(len(X_)):
                    if X_[p]==0:
                        prod*=self.ph0_j[k][p]
                    else :
                        prod*=self.ph1_j[k][p]
                real_prob.append(prod)
            probab.append(real_prob)
        return probab
    
    def fit(self, X, y):
        self.m = len(y)
        self.n = X.shape[1]
        
        self.labels = np.unique(y)
        
        self.calculate_phi_s(y)
        
        self.compute_phi_x0y(X, y)
        self.compute_phi_x1y(X, y)
        
    def predict(self, X):
        probs = np.array(self.prob_x_y(X))
        
        pred = probs.T * self.phi
        
        
        return self.labels[np.argmax(pred, axis=1)]
        
        
    def accuracy(self, y, y_hat):
        count=0
        for i in range(len(y)):
            if y[i]==y_hat[i]:
                count+=1
        return count/len(y)