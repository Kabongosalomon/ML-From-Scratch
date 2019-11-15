import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, num_iter=100000, batch_size=1, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        
        X = self.__add_intercept(X)
        
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
                        
            rand = np.random.choice(y.size, self.batch_size).squeeze()
            gradient = np.dot(X[rand].T, (h[rand] - y[rand]))/y.size   
        
            self.theta -= self.lr * gradient
            
            if(self.verbose == True and i % 100 == 0):
                z = np.dot(X, self.theta)
                h = self.__sigmoid(z)
                print(f'loss: {self.__loss(h, y)} \t')
    
    def predict_prob(self, X):
        X = self.__add_intercept(X)
    
        return self.__sigmoid(np.dot(X, self.theta))
    
    def predict(self, X, threshold=.5):
        return self.predict_prob(X) >= threshold