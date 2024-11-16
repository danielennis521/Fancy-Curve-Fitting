import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


class PolyLeastSquares():
    def __init__(self, degree=1, learning_rate=2*1e-6, max_iterations=2*int(1e6), tol=None, momentum=0.2):
        self.degree = degree
        self.n = degree+1
        self.a = np.zeros(degree+1)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.gamma = momentum
        

    def fit(self, x, y, method='normal'):
        if self.tol is None:
            self.tol = 1e-1 * la.norm(y)

        v = np.zeros(self.n)
        self.x = x
        self.y = y
        self.a = np.zeros(self.n)
        self.m = len(x)
        self.X = np.array([np.power(x, i) for i in range(self.n)]).T

    

        functions = {'normal': self.fit_normal
                     , 'gd': self.fit_gd
                     , 'momentum': self.fit_momentum
                     , 'nesterov': self.fit_nesterov}

        return functions[method](x, y)

    # "exact" solution via normal equations
    def fit_normal(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

        A = np.array([x**i for i in range(self.n)]).transpose()

        self.a = la.lstsq(A, y,rcond=None)[0]
         # correcting for errors due to numerical precision
        for j in range(self.n):
            if abs(self.a[j]) < 1e-10: self.a[j] = 0

        return self.a

    # standard gradient descent
    def fit_gd(self, x, y):
        
        for i in range(self.max_iterations):
            error = np.dot(self.X, self.a) - y
            gradient = np.dot(self.X.T, error)
            self.a -= self.learning_rate * gradient * (1/self.m)

            if la.norm(gradient) < self.tol or self.a[0] > 1e6:
                break

        return self.a

    # gradient descent with momentum
    def fit_momentum(self, x, y):

        for i in range(self.max_iterations):
            error = np.dot(self.X, self.a) - y
            gradient = np.dot(self.X.T, error)
            v = self.gamma*v + self.learning_rate * gradient * ((1-self.gamma)/self.m)
            self.a -= v

            if la.norm(gradient) < self.tol or self.a[0] > 1e6:
                break

        return self.a

    # Nesterov accelerated gradient descent
    def fit_nesterov(self, x, y):     
        
        for i in range(self.max_iterations):
            error = np.dot(self.X, self.a+self.gamma*v) - y
            gradient = np.dot(self.X.T, error)
            v = self.gamma*v + self.learning_rate * gradient * ((1-self.gamma)/self.m)
            self.a -= v

            if la.norm(gradient) < self.tol or self.a[0] > 1e6:
                break

        return self.a
    

    def plot_Data(self, data_marker='ro'):
        plt.plot(self.x, self.y, data_marker)
        plt.show()
        plt.cla()
        return


    def plot_Fit(self, fit_marker='b'): 
        x = np.linspace(min(self.x), max(self.x), 100)
        y = np.sum([self.a[i]*x**i for i in range(self.n)], axis=0)
        plt.plot(x, y, fit_marker)
        plt.show()
        plt.cla()
        return
    

    def plot_Both(self, data_marker='ro', fit_marker='b'):
        x = np.linspace(min(self.x), max(self.x), 100)
        y = np.sum([self.a[i]*x**i for i in range(self.n)], axis=0) 
        plt.plot(self.x, self.y, data_marker, x, y, fit_marker)
        plt.show()
        plt.cla()       
        return
