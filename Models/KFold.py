import numpy as np 
import ipdb

class KFoldXV():
    def __init__(self, folds=5, regr_val = 0) :
        self.folds = folds
        self.regr_val = regr_val
    """
    Input :
    -----
    data : input dataset as tupple (X,y)
    model : the model used on the Data
    
    loss : the loss function 
    folds : the number of k
    
    Return :
    ------
    """
    def fit(self, data, model):
        X = data[0]
        X_ = X.copy()
        y = data[1]
        y_ = y.copy()
        
        # Split D in to k mutaully exclusive subsets(Di), which union is D
        Di={}
        loss = []
        k_m = int(np.floor(X.shape[0]/self.folds))
        for i in range(self.folds-1):
            rand = np.random.choice(X_.shape[0],k_m, replace=False)
            Di[i]=rand
            X_ = np.delete(X_, rand, 0)
            ipdb.set_trace()
        Di[self.folds-1]= np.random.choice(X_.shape[0],X_.shape[0])
        for i in range(self.folds):
            model.fit(np.delete(X, Di[i], 0), np.delete(y, Di[i], 0))
            y_hat =  model.predict(X[rand])
            if self.regr_val == 0 :
                loss.append(((y[rand]-y_hat)**2).mean())
            elif self.regr_val == 1:
                loss.append((y[rand] == y_hat).mean())
            
        return np.array(loss).mean()