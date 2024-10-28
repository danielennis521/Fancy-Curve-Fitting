from numpy import array, zeros, sign, abs, linspace, sum, dot, insert
from numpy import linalg as la
import matplotlib.pyplot as plt


class PolyLeastSquares():
    def __init__(self, degree=1):
        self.n = degree+1
        self.a = zeros(degree+1)
        

    def fit(self, x, y):
        self.x = array(x)
        self.y = array(y)

        A = array([x**i for i in range(self.n)]).transpose()

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
        x = linspace(min(self.x), max(self.x), 100)
        y = sum([self.a[i]*x**i for i in range(self.n)], axis=0)
        plt.plot(x, y, 'b')
        plt.show()
        return
    
    def plot_Both(self):
        x = linspace(min(self.x), max(self.x), 100)
        y = sum([self.a[i]*x**i for i in range(self.n)], axis=0) 
        plt.plot(self.x, self.y, 'ro', x, y, 'b')
        plt.show()       
        return
    

class L1LinearRegression():
    def __init__(self, learning_rate=0.01, max_iterations=1000, L1_penalty=1, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.L1_penalty = L1_penalty
        self.tol = tol
        self.a = array([])
        self.n = 0  # number of observations
        self.m = 0  # number of regressors

    def fit(self, x, y):
        
        self.x = array(x)
        self.x = insert(self.x, 0, 1, axis=1)
        self.y = array(y)
        self.n, self.m = self.x.shape
        self.a = zeros(self.m)

        for i in range(self.max_iterations):
            a = self.a.copy()
            self.a -= self.learning_rate * self.gradient()/self.n

            if sum(abs(a - self.a)) < self.tol: break

        return self.a
        
    def gradient(self):
        penalty = self.L1_penalty * sign(self.a)
        penalty[0] = 0
        return -2 * dot(self.x.T, self.y - dot(self.x, self.a)) + penalty

test.fit([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
print(test.a)
print(sign(0))


class L1PolyLeastSquares():
    def __init__(self, degree=1, L1_penalty=1, learning_rate=0.01, max_iterations=1000, tol=1e-6):
        self.n = degree+1
        self.a = np.zeros(degree+1)
        self.L1_penalty = L1_penalty
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tol = tol

    def fit(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.n = len(x)
        self.a = np.zeros(self.n)

        X = np.array([x**i for i in range(self.n)]).transpose()
        model = L1LinearRegression(self.learning_rate, self.max_iterations, self.L1_penalty, self.tol)
        self.a = np.flip(model.fit(X, y))

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
