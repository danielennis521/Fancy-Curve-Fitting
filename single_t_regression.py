# Normally linear regression is done by minimizing the residual summ of squares.
# Doing this gives us the maximum Liklehood estimate (MLE) of the parameters that
# make up the regression function if the residuals are normally distributed.
# The code here is for an alternative case, namely, when the residuals follow a 
# student-t distribution. This leads to a set of non-linear equations and we are forced
# to resort iterative methods to estimate the regression parameters. The student-t 
# distribution has much heavier tails than the normal distribution, this leads to the 
# regression based on student-t residuals being less sensitive to outliers, with the 
# level of sensativity being controlled by the degrees of freedom.

import numpy as np



class single_tRegression():
    def __init__(self, dof=-1, learning_rate=1e-3, max_steps=int(1e4),
                  tol=0.0, solver='gradient'):

        methods = {'gradient': self.grad_step}
        self.dof = dof
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.tol = tol

    def fit(self, x, y):
        self.x = x
        self.y = y
        self.a = 0.0
        self.b = 0.0
        ap = 0.0
        bp = 0.0

        for i in range(self.max_steps):
            self.grad_step()

            change = (ap - self.a)**2 + (bp - self.b)**2
            if change <= self.tol:
                break


    def grad_step(self):
        ga = 0.0
        gb = 0.0
        for i in range(len(self.x)):
            t = (self.y[i] - self.a - self.b*self.x[i])
            ga += t / (self.dof + t**2)
            gb += t*self.x[i] / (self.dof + t**2)

        self.a += self.learning_rate * (self.dof+1)*ga
        self.b += self.learning_rate * (self.dof+1)*gb    


    def predict(self, x):
        return self.a + self.b * x