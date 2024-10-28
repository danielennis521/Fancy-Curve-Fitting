from numpy import array, zeros, sign, abs
from numpy import linalg as la
import matplotlib.pyplot as plt

class PolyLeastSquares():
    def __init__(self, degree=1):
        self.n = degree+1
        self.a = zeros(degree+1)
        

    def fit(self, x, y):
        self.x = array(x)
        self.y = array(y)

        w = [sum(self.x**i) for i in range(2*self.n)]
        A = array([w[i:i+self.n] for i in range(self.n)])
        b = array([sum(self.y*self.x**i) for i in range(self.n)])

        self.a = la.solve(A, b)
         # correcting for errors due to numerical precision
        for j in range(self.n):
            if abs(self.a[j]) < 1e-12: self.a[j] = 0

        return self.a       

    def plot_Data():

        return

    def plot_Fit(): 

        return
    
    def plot_Both():

        return
    


class L1LeastSquares():
    def __init__(self, degree=1, penalty=0.1, learningRate=0.01, max_iter=100, tol=1e-3):
        self.n = degree+1
        self.a = zeros(degree+1)
        self.penalty = penalty
        self.learningRate = learningRate
        self.max_iter = max_iter
        self.tol = tol
        self.A = array([])
        self.b = array([])
        self.initial_guess = PolyLeastSquares(self.n-1)

    def fit(self, x, y):
        
        self.a = self.initial_guess.fit()
        self.x = array(x)
        self.y = array(y)

        w = [sum(self.x**i) for i in range(2*self.n)]
        A = array([w[i:i+self.n] for i in range(self.n)])
        b = array([sum(self.y*self.x**i) for i in range(self.n)])

        for i in range(self.max_iter):
            a = self.a - self.learningRate * self.gradient()
             # correcting to help ensure convergence
            for j in range(self.n):
                if abs(a[j]) < 1e-12: a[j] = 0
            if la.norm(a - self.a) < self.tol: break
            self.a = a
        return self.a

    def gradient(self):
        return 2*(la.multi_dot([self.A, self.a]) - self.b) + self.penalty*sign(self.a)
    
    def plot_Data():

        return

    def plot_Fit(): 

        return
    
    def plot_Both():

        return



test = PolyLeastSquares(2)

test.fit([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
print(test.a)
print(sign(0))
