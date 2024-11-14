import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt


class PolyLeastSquares():
    def __init__(self, degree=1):
        self.n = degree+1
        self.a = np.zeros(degree+1)
        

    def fit(self, x, y):


    def fit_normal(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)

        A = np.array([x**i for i in range(self.n)]).transpose()

        self.a = la.lstsq(A, y,rcond=None)[0]
         # correcting for errors due to numerical precision
        for j in range(self.n):
            if abs(self.a[j]) < 1e-10: self.a[j] = 0

        return self.a

    def fit_gd(self, x, y, degree=1, learning_rate=1e-3, max_iterations=1000, tol=None)
        if tol is None:
            tol = 1e-2 * la.norm(y)

        self.x = x
        self.y = y
        self.a = np.zeros(self.n)
        self.m = len(x)
        X = np.array([np.power(x, i) for i in range(self.n)]).T
        
        for i in range(self.max_iterations):
            error = np.dot(X, self.a) - y
            gradient = np.dot(X.T, error)
            self.a -= self.learning_rate * gradient * (1/self.m)

            if la.norm(gradient) < self.tol or self.a[0] > 1e6:
                break

        print(la.norm(gradient))
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
    

    
