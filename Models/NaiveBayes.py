import numpy as np
from numpy.linalg import inv
import scipy.stats


class NaiveBayes:
    def __init__(self):
        self.phi = []
        self.phi_j = []
        self.prob_product = None
        self.labels = None
        self.m = None
        self.n = None
    
    def calculate_phi(self, y):
        for i in range(len(self.labels)):
            self.phi.append((sum(y[y==self.labels[i]])+1)/(len(y)+len(self.labels)))
#             self.phi.append((y[y==self.labels[i]]).mean())
    
    def calculate_phi_j(self, x, y, j):
        for i in range(len(self.labels)):
            self.phi_j.append(np.log(((x[i][j] == 1 and y[i] == self.labels[i])+1)/(sum(y==self.labels[i])\
                                                                             +len(self.labels))))   
            
#             self.phi_j.append(np.log1p(((x[i][j] == 1 and y[i] == self.labels[i]))/(sum(y==self.labels[i]))))
        
    def fit(self, X, y):
        self.m = len(y)
        self.n = X.shape[1]
        
        self.labels = np.unique(y)
        
        self.calculate_phi(y)
        self.prob_product = self.prob_xy(X, y)
        
    
    def prob_xy(self, X, y):
        for j in range(self.n):
            self.calculate_phi_j(X, y, j)
            
            
    def predict(self, X):
        total = []
        prob_product = sum(self.phi_j)
#         prob_product = np.prod(self.phi_j)
        denominator = sum([prob_product*self.phi[i] for i in range(len(self.labels))])
        for i_ in range(self.m):
            result = []
            for i in range(len(self.labels)):
                
                result.append(prob_product * self.phi[i]/denominator)
            total.append(np.argmax(result))
        return np.array(total)
    
    def predict_prob(self, X):
        total = []
        prob_product = sum(self.phi_j)
        denominator = sum([prob_product*self.phi[i] for i in range(len(self.labels))])
        for i_ in range(self.m):
            result = []
            for i in range(len(self.labels)):
                
                result.append(prob_product * self.phi[i]/denominator)
            total.append(result)
        return np.array(total)
    
    def accuracy(self, y, y_hat):
        count=0
        for i in range(len(y)):
            if y[i]==y_hat[i]:
                count+=1
        return count/len(y)