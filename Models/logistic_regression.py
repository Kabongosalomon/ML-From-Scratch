import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.001, num_iter=100000, batch_size=1, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.batch_size = batch_size
        self.verbose = verbose
        self.h = None
        self.labels = None
        self.n_label = None
        self.thetas = []
        
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def __sofmax(self, z):
        exp = np.exp(z-np.max(z, axis=1).reshape((-1,1)))
        norms = np.sum(exp, axis=1).reshape((-1,1))
        return exp / norms
    
    def __loss(self, h, y):
        return (-y * np.log(h)).sum()
#         return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, X, y):
        
        self.labels = np.unique(y)
        # To handle the case where the y is a float
        self.labels = self.labels.astype('i')        
        self.n_label = len(self.labels)
        
        X = self.__add_intercept(X)
        
        # weights initialization
        
        
        for k in range(self.n_label):
            theta = np.zeros(X.shape[1])
            for i in range(self.num_iter):
                z = np.dot(X, theta)
                h = self.__sigmoid(z)

                y_ = np.where(y == k, 1, 0)
    
                rand = np.random.choice(y_ .size, self.batch_size).squeeze()
                gradient = np.dot(X[rand].T, (h[rand] - y_ [rand]))/y_ .size   

                theta -= self.lr * gradient

                if(self.verbose == True and i % 100 == 0):
                    z = np.dot(X, theta)
                    h = self.__sigmoid(z)
                    print(f'loss: {self.__loss(h, y_ )} \t')
            self.thetas.append(theta)
    
    def predict_prob(self, X):
        X = self.__add_intercept(X)

        return self.__sofmax(np.dot(X, np.array(self.thetas).T))
    
    def predict(self, X,):
        return self.labels[np.argmax(self.predict_prob(X), axis=1)]
    
    def accuracy(self, y, y_hat):
        count=0
        for i in range(len(y)):
            if y[i]==y_hat[i]:
                count+=1
        return count/len(y)

    np.linalg.qr