#coding:utf-8

import numpy as np

class LogisticRegression:

    def __init__(self, tol = 1e-4, max_itr=300, beta=None):
        '''
            init function of LogisticRegression

            @tol: float, tolerance for stopping criteria

            @max_itr: int, max iterations

            @beta: np.array/list, shape: [n_features,]
        '''
        self._tol = tol
        self._max_itr = max_itr
        self._beta = np.matrix(beta).T
        self._fityet = False
    def fit(self, X,y):
        '''
            fit model using test data X and label y

            @X: np.ndarray, shape: [n_samples, n_features]
            
            @y: np.ndarray, ground truth of X, shape: [n_samples, ]

            #return: void 
        '''
        #init
        X = np.array(X)
        y = np.array(y)
        X = np.matrix(np.hstack((X, np.ones((X.shape[0],1))))).T # shape: [n_features, n_sample]
        y = np.matrix(y).T # shape: [n_samples, 1]
        d = X.shape[0]
        p_1_func = lambda X, beta: 1/(1+np.exp(-X.T*beta))
        self._beta = np.matrix(np.zeros((d,1))) if self._beta is None else self._beta
        if self._beta.shape[0] != d:
            raise Exception('beta dimension error')
        #newton iteration
        itrs = 0
        while itrs < self._max_itr:
            itrs += 1
            p_1 = p_1_func(X, self._beta)
            df = -1 * X*(y-p_1)
            ddf = np.matrix(np.zeros((d,d)))
            for i in xrange(X.shape[1]):
                ddf += (p_1[i,:]*(1-p_1[i,:]))[0,0] * X[:,i] * X[:,i].T 
            diff = np.linalg.pinv(ddf) * df
            if np.linalg.norm(diff) < self._tol:
                break
            self._beta -= diff
        self._fityet = True
    
    def predict(self, X, prob=False):
        '''
            predict label of X, must be invoked after fit()

            @X: np.ndarray, shape: [n_samples, n_features]

            @prob: bool, whether return probability or label

            #return: np.ndarray(), shape: [n_samples,]
        '''
        if self._beta is None:
            raise Exception('fit not invoked yet')
        X = np.array(X)
        X = np.matrix(np.hstack((X,np.ones((X.shape[0],1))))).T
        sigmod_v = 1.0/(1.0+np.exp(-self._beta.T*X))
        if prob:
            return np.array(sigmod_v.flatten().tolist()[0])
        else:
            label_res = (sigmod_v.flatten()>=0.5).astype(int)
            return np.array(label_res.tolist()[0])
