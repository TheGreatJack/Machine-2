import numpy as np
import matplotlib.pyplot as plt
import Utils.mlutils as mlutils
from scipy.optimize import minimize

class SLP:
    def __init__(self, h_units, reg=0, init_params=None):
        self.h_units = h_units
        self.reg     = reg
        self.init_params = init_params

    def sigm(self, x):
        return 1/(1+np.exp(-x))

    def encode(self, b1, b2, W1, W2): 
        return np.array(list(b1)+[b2]+list(W1.flatten())+list(W2.flatten()))

    def decode(self, params, nb_cols):
        b1 = params[:self.h_units]
        b2 = params[self.h_units]
        t = self.h_units+1+nb_cols*self.h_units
        W1 = params[self.h_units+1:t].reshape(nb_cols, self.h_units)
        W2 = params[t:].reshape(-1,1)
        return b1, b2, W1, W2

    def y_hat(self, X, params):
        b1,b2,W1,W2 = self.decode(params, X.shape[1])
        return self.sigm(np.tanh(X.dot(W1)+b1).dot(W2)+b2)[:,0]

    def k(self, X, y, params):
        b1,b2,W1,W2 = self.decode(params, X.shape[1])
        return (self.sigm(np.tanh(X.dot(W1)+b1).dot(W2)+b2)-y)
    
        
    def fit(self, X,y, verbose=False):

        self.init_params = np.random.random(1+self.h_units*2+self.h_units*X.shape[1]) \
                           if self.init_params is None else self.init_params
        
        def cost(p):
            return np.mean ( (self.y_hat(X, p) - y)**2 ) + self.reg * np.sum(p**2)

        r = minimize(cost, self.init_params, method="BFGS")
        self.params = r.x

    def predict(self, X):
        return (self.y_hat(X, self.params)>.5)*1

    def score(self, X,y):
        return np.sum(self.predict(X)==y)*1./len(X)

    def draw(self, ax=None):
        ax = plt.figure(figsize=(4,4)).add_subplot(111) if not ax else ax
        mlutils.draw_neural_net(ax, .1, .9, .1, .9, [2, self.h_units, 1])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.axis("off")