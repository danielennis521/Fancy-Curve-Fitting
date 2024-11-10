import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


class PolyLeastSquares():
    def __init__(self, degree=1):
        self.n = degree+1
        self.a = np.zeros(degree+1)
        

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

        A = np.array([x**i for i in range(self.n)]).transpose()

        self.a = la.lstsq(A, y,rcond=None)[0]
         # correcting for errors due to numerical precision
        for j in range(self.n):
            if abs(self.a[j]) < 1e-10: self.a[j] = 0

        return self.a       

    def plot_Data(self):
        plt.plot(self.x, self.y, 'ro')
        plt.show()
        return

    def plot_Fit(self): 
        x = np.linspace(min(self.x), max(self.x), 100)
        y = np.sum([self.a[i]*x**i for i in range(self.n)], axis=0)
        plt.plot(x, y, 'b')
        plt.show()
        return
    
    def plot_Both(self):
        x = np.linspace(min(self.x), max(self.x), 100)
        y = np.sum([self.a[i]*x**i for i in range(self.n)], axis=0) 
        plt.plot(self.x, self.y, 'ro', x, y, 'b')
        plt.show()       
        return
    

class L1LinearRegression():
    def __init__(self, learning_rate=1e-6, max_iterations=1000, L1_penalty=1, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.L1_penalty = L1_penalty
        self.tol = tol
        self.a = np.array([])
        self.n = 0  # number of observations
        self.m = 0  # number of regressors

    def fit(self, x, y, a=None):
        
        self.x = np.array(x)
        self.x = np.insert(self.x, 0, 1, axis=1)
        self.x_norm = self.normalize(self.x.copy())
        self.y = np.array(y)
        self.n, self.m = self.x.shape
        if a is None:
            self.a = np.zeros(self.m)
        else:
            self.a = a

        for i in range(self.max_iterations):
            a = self.a.copy()
            self.a -= self.learning_rate * (1/self.n) * self.gradient()
            if sum(abs(a - self.a)) <= self.tol: break

        return a
        
    def gradient(self):
        penalty = self.L1_penalty * np.sign(self.a)
        penalty[0] = 0
        return np.dot(self.x_norm.T, np.dot(self.x, self.a) - self.y) + 0.5*penalty

    def normalize( self, X ) :
        print(np.mean( X[:, 1:], axis = 0 ))
        print(np.std( X[:, 1:], axis = 0 ))
        X[:, 1:] = ( X[:, 1:] - np.mean( X[:, 1:], axis = 0 ) ) / np.std( X[:, 1:], axis = 0 )
        return X


class L1PolyLeastSquares():
    def __init__(self, degree=1, L1_penalty=1, learning_rate=1e-5, max_iterations=int(1e4), tol=1e-6):
        self.n = degree+1
        self.a = np.zeros(degree+1)
        self.L1_penalty = L1_penalty
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.a = np.zeros(self.n)

        lsq = PolyLeastSquares(self.n-1)
        initial_guess = lsq.fit(x, y)

        X = np.array([np.power(x, i) for i in range(1, self.n)]).transpose()

        model = L1LinearRegression(self.learning_rate, self.max_iterations, self.L1_penalty, self.tol)
        self.a = model.fit(X, y)

        return self.a
    
    def plot_Data(self):
        plt.plot(self.x, self.y, 'ro')
        plt.show()
        return

    def plot_Fit(self): 
        x = np.linspace(min(self.x), max(self.x), 100)
        y = np.sum([self.a[i]*x**i for i in range(self.n)], axis=0)
        plt.plot(x, y, 'b')
        plt.show()
        return
    
    def plot_Both(self):
        x = np.linspace(min(self.x), max(self.x), 100)
        y = np.sum([self.a[i]*x**i for i in range(self.n)], axis=0) 
        plt.plot(self.x, self.y, 'ro', x, y, 'b')
        plt.show()       
        return
    
