#coding:utf-8

import numpy as np

class FullyConnectedNN:

    def __init__(self, layer_sizes, activation_func, derivative_func, tol=1e-3):
        '''
            init function of Fully Connected Neural Network

            @layer_sizes: np.ndarray, shape=(n,) where n>=2
            @active_func: callable, activation function
            @derivative_func: callable, derivative function
            @tol: float, tolerance to stop the gradient descent  
        '''
        self._layer_sizes = layer_sizes
        self._w_mats = [] 
        self._b_mats = []
        self._act_func = activation_func
        self._der_func = derivative_func
        self._tol = tol
        self._fitted = False

    def init_w_b(self):
        '''
            init self._w_mats and self._b_mats
        '''
        #init with normal
        for l in xrange(0, self._layer_sizes.shape[0]-1):
            w_l = np.matrix(np.random.normal(size=(self._layer_sizes[l+1], self._layer_sizes[l]), scale=0.01))
            b_l = np.matrix(np.random.normal(size=(self._layer_sizes[l+1], 1), scale=0.01))
            self._w_mats.append(w_l)
            self._b_mats.append(b_l)

    def fit(self,X,Y, alpha, lamb):
        '''
            fit X and Y with learning rate alpha and regularization factor lamb

            @X: np.ndarray, shape=(n_samples, n_features)
            @Y: np.ndarray, shape=(n_samples, n_output_size)
            @alpha: float, learning rate
            @lamb: float, regularization factor
        '''
        #transform X and Y to matrix
        X = np.matrix(X).T
        Y = np.matrix(Y).T
        #init
        self.init_w_b()
        #gradient descent 
        last_loss = np.inf
        while True:
            #initial delta_w matrix and delta_b vector
            step_w_mats = [
                np.matrix(np.zeros((self._layer_sizes[i+1], self._layer_sizes[i]))) for i in xrange(self._layer_sizes.shape[0]-1)
            ]
            step_b_mats = [
                np.matrix(np.zeros((self._layer_sizes[i+1], 1))) for i in xrange(self._layer_sizes.shape[0]-1)
            ]
            #backpropagation
            tmp_loss = 0.
            for i in xrange(X.shape[1]):
                x = X[:,i]
                y = Y[:,i]
                z_vecs = [x]
                a_vecs = [x]
                #forward
                for l in xrange(self._layer_sizes.shape[0]-1):
                    z_vecs.append(self._w_mats[l]*a_vecs[l]+self._b_mats[l])
                    a_vecs.append(self._act_func(np.array(z_vecs[l+1])))
                #loss
                tmp_loss += ((y-a_vecs[-1]).T*(y-a_vecs[-1])/2)[0,0]
                #backward
                residual_errors = [ np.multiply(self._der_func(z_vecs[-1]), a_vecs[-1]-y) ]
                step_w_mats[-1] += residual_errors[-1] * a_vecs[-2].T
                step_b_mats[-1] += residual_errors[-1]
                for i in xrange(len(self._w_mats)-1, 0, -1):
                    re = np.multiply(self._w_mats[i].T * residual_errors[0], self._der_func(np.array(z_vecs[i])))
                    residual_errors.insert(0, re)
                    step_w_mats[i-1] += residual_errors[0] * a_vecs[i-1].T
                    step_b_mats[i-1] += residual_errors[0]
            
            for i in xrange(len(self._w_mats)):
                self._w_mats[i] -= alpha*( step_w_mats[i]/X.shape[1] + lamb*self._w_mats[i] )
                self._b_mats[i] -= alpha*( step_b_mats[i]/X.shape[1] )
            #check whether to stop the iteration
            if np.abs(last_loss-tmp_loss) <= self._tol:
                break
            else:
                last_loss = tmp_loss
        self._fitted = True

    def predict(self, X):
        '''
            predict for X

            @X: np.ndarray, shape=(n_samples, n_features)

            #Y: np.ndarray, shape=(n_samples, n_output_size)
        '''
        X = np.matrix(X).T
        Y = []
        for i in xrange(X.shape[1]):
            x = X[:,i]
            z_vecs = [x]
            a_vecs = [x]
            #forward
            for l in xrange(self._layer_sizes.shape[0]-1):
                z_vecs.append(self._w_mats[l]*a_vecs[l]+self._b_mats[l])
                a_vecs.append(self._act_func(np.array(z_vecs[l+1])))
            Y.append(a_vecs[-1])
        return np.array(Y)

def sigmod(x):
    '''
        sigmod activation function

        @x: np.ndarray, shape=(n_samples,)

        #y: np.ndarray, shape=(n_samples,)
    '''
    x = np.array(x)
    return 1 / (1 + np.exp(-1*x))

def derivative_sigmod(x):
    '''
        derivative function of sigmod function

        @x: np.ndarray, shape=(n_samples,)

        #y: np.ndarray, shape=(n_samples,)
    '''
    s = sigmod(np.array(x))
    return s * (1 - s)
