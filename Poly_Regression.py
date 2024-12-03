import numpy as np
import numpy.polynomial.polynomial as p
from numpy import linalg as la
import matplotlib.pyplot as plt
import polynomial_basis as b


class PolyLeastSquares():
    def __init__(self, degree=1, learning_rate=2*1e-6, max_iterations=int(1e6), tol=None, momentum=0.2
                 , basis='monomial', normalize=False):
        # inputs:
        # degree: degree of polynomial used in regression
        # learning_rate: step size when using gradient descent
        # max_iterations: maximum number of iterations for any gradient descent variant
        # tol: tolerance for stopping criteria
        # momentum: momentum parameter for gradient descent with momentum
        #           0 <= momentum < 1, 0 is standard gradient descent
        # basis: class of polynomials used as regressors\
        #        options: monomial, chebyshev, legendre, hermite
        # normalize: boolean, if True, regressors are normalized to z-scores

        self.degree = degree
        self.n = degree+1
        self.c = np.zeros(degree+1)
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol
        self.gamma = momentum
        self.basis = basis  
        self.normalize = normalize
        

    def fit(self, x, y, method='numpy'):
        # inputs:
        # x: array of points where data was collected
        # y: array of observed values at points x
        # method: method used to solve for coefficients
        #         options: numpy, normal, gd, momentum, nesterov
        #
        # returns: array of coefficients for polynomial fit

        if self.tol is None:
            self.tol = 1e-1 * la.norm(y)

        v = np.zeros(self.n)
        self.x = x
        self.y = y
        self.c = np.zeros(self.n)
        self.m = len(x)

        try:
            self.L = self.generate_basis()
        except ValueError as e:
            print(e)
            return

        A = [[p.polyval(z, l) for l in self.L] for z in self.x]
        self.X = np.array(A)

        if self.normalize:
            self.convert_to_zscores()

        functions = {'numpy': self.fit_numpy
                     , 'normal': self.fit_normal
                     , 'gd': self.fit_gd
                     , 'momentum': self.fit_momentum
                     , 'nesterov': self.fit_nesterov}

        return functions[method]()


    # uses numpy's lstsq function to solve for least squares coefficients
    def fit_numpy(self):
        self.c = la.lstsq(self.X, self.y,rcond=None)[0]
         # correcting for errors due to numerical precision
        for j in range(self.n):
            if abs(self.c[j]) < 1e-10: self.c[j] = 0

        return self.c
    
    
    # "exact" solution via normal equations
    def fit_normal(self):
        A = np.dot(self.X.T, self.X)
        b = np.dot(self.X.T, self.y)
        self.c = la.solve(A, b)
        for j in range(self.n):
            if abs(self.c[j]) < 1e-10: self.c[j] = 0

        return self.c


    # standard gradient descent
    def fit_gd(self):
        
        for i in range(self.max_iterations):
            error = np.dot(self.X, self.c) - self.y
            gradient = np.dot(self.X.T, error)
            self.c -= self.learning_rate * gradient * (1/self.m)

            if la.norm(gradient) < self.tol or self.c[0] > 1e6:
                break

        return self.c


    # gradient descent with momentum
    def fit_momentum(self):

        for i in range(self.max_iterations):
            error = np.dot(self.X, self.c) - self.y
            gradient = np.dot(self.X.T, error)
            v = self.gamma*v + self.learning_rate * gradient * ((1-self.gamma)/self.m)
            self.c -= v

            if la.norm(gradient) < self.tol or self.c[0] > 1e6:
                break

        return self.c


    # Nesterov accelerated gradient descent
    def fit_nesterov(self):     
        
        for i in range(self.max_iterations):
            error = np.dot(self.X, self.c+self.gamma*v) - self.y
            gradient = np.dot(self.X.T, error)
            v = self.gamma*v + self.learning_rate * gradient * ((1-self.gamma)/self.m)
            self.c -= v

            if la.norm(gradient) < self.tol or self.c[0] > 1e6:
                break

        return self.c
    

    def generate_basis(self):
        if self.basis == 'monomial':
            return b.monomials(self.n)
        elif self.basis == 'chebyshev':
            return b.Chebyshev_Polynomials(self.n, a=min(self.x), b=max(self.x))
        elif self.basis == 'legendre':
            return b.Legendre_Polynomials(self.n, a=min(self.x), b=max(self.x))
        elif self.basis == 'hermite':
            return b.Hermite_Polynomials(self.n)
        else:
            raise ValueError('Invalid basis type, please choose from: monomial, chebyshev, legendre, hermite')


    def convert_to_zscores(self):
        mu = np.mean(self.X[:, 1:], axis = 0)
        sigma = np.std(self.X[:, 1:], axis = 0) 
        self.X[:, 1:] = (self.X[:, 1:] - mu) / sigma
        self.L[1:] = [p.polymul(1/s, p.polyadd(-m, l)) for m, s, l in zip(mu, sigma, self.L[1:])]
        return


    def get_coefficients(self):
        c = [0]
        for i in range(self.n):
            c = p.polyadd(c, p.polymul([self.c[i]], self.L[i]))
        return c


    def predict(self, x):
        A = [[p.polyval(x, l) for l in self.L] for x in x]
        return np.dot(A, self.c)
    

    def plot_Data(self, data_marker='ro'):
        plt.plot(self.x, self.y, data_marker)
        plt.show()
        plt.cla()
        return


    def plot_Fit(self, fit_marker='b'): 
        x = np.linspace(min(self.x), max(self.x), 100)
        y = self.predict(x)
        plt.plot(x, y, fit_marker)
        plt.show()
        plt.cla()
        return
    

    def plot_Both(self, data_marker='ro', fit_marker='b'):
        x = np.linspace(min(self.x), max(self.x), 100)
        y = self.predict(x)
        plt.plot(self.x, self.y, data_marker, x, y, fit_marker)
        plt.show()
        plt.cla()       
        return
    
