#coding:utf-8

import numpy as np

class LogisticRegression:

    def __init__(self, tol = 1e-4, max_itr=300):
        '''
            init function of LogisticRegression

            @tol: float, tolerance for stopping criteria
            @max_itr: int, max iterations
        '''
        self._tol = tol
        self._max_itr = max_itr
        self._beta = None
    def fit(self, X,y):
        '''
            fit model using test data X and label y

            @X: np.ndarray, shape: [n_samples, n_features]
            
            @y: np.ndarray, ground truth of X, shape: [n_samples, ]

            #return: void 
        '''
        X = np.matrix(np.hstack((X, np.ones((X.shape[0],1))))).T # shape: [n_features, n_sample]
        y = np.matrix(y).T # shape: [n_samples, 1]

        d = X.shape[0]
        p_1_func = lambda X, beta: 1/(1+np.exp(-X.T*beta))
        
        self._beta = np.matrix(np.zeros((d,1)))

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
    
    def predict(self, x):
        '''
            predict label of x, must be invoked after fit()

            @x: np.ndarray, shape: [n_features, ]

            #return: 0/1
        '''

        x = np.matrix(np.hstack((x,np.array([1])))).T
        sigmod_v = 1.0/(1.0+np.exp(-self._beta.T*x))
        return 1 if sigmod_v >= 0.5 else 0


X = np.array([[0.5,0.5],[1.,1.],[5.,5.5],[4.5,4.]])
y = np.array([1,1,0,0])

lr = LogisticRegression()
lr.fit(X,y)
label = lr.predict(np.array([50.,41.]))
print label    

