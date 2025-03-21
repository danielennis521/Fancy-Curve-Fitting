import numpy as np



def Lm_Poly_Regression(x, y, degree=2, m=2, steps=int(1e5), alpha=0.5, learning_rate=1e-3, tol=0.0):
    # Extends the idea of regularized polynomial regression to Lm
    # L1 is Lasso and L2 is ridge regression. 
    #
    # inputs:
    # x: points where observations were made
    # y: observations
    # degree: degree of the polynomial to be fit to the data
    # steps: maximum number of steps to be used in gradient descent
    # alpha: how much weight to give the regularization term
    # learning_rate: step size for gradient descent
    # tol: minimum gradient stopping condition
    #
    # outputs:
    # coefficients of the polynomial fit starting with the constant term / bias

    M = np.array([x**(i+1) for i in range(degree)]).T
    n = len(x)
    a = np.zeros(degree)
    b = 0

    for i in range(steps):
        predicted = M.dot(a) + b
        
        if m==0:
            grad_a = -2*(y - predicted).dot(M)/n
            grad_b = -2*np.sum(y - predicted)/n
        elif m==1:
            grad_a = -2*(y - predicted).dot(M)/n + alpha*np.sign(a)/n
            grad_b = -2*np.sum(y - predicted)/n
        else:
            grad_a = -2*(y - predicted).dot(M)/n + alpha*m*np.abs(a**(m-1))*np.sign(a)/n
            grad_b = -2*np.sum(y - predicted)/n

        if np.linalg.norm(learning_rate*grad_a)/n <= tol:
            break

        a = a - learning_rate*grad_a
        b = b - learning_rate*grad_b

    return np.concatenate([[b], a])



def Taylor_Net(x, y, degree=2, m=[1, 2], steps=int(1e5), alpha=[0.5, 0.5], learning_rate=1e-3, tol=0.0):

    M = np.array([x**(i+1) for i in range(degree)]).T
    n = len(x)
    a = np.zeros(degree)
    b = 0


    for i in range(steps):
        predicted = M.dot(a) + b
        grad_a = -2*(y - predicted).dot(M)/n
        grad_b = -2*np.sum(y - predicted)/n

        for j in range(len(m)):
            if m[j]==1:
                grad_a += alpha[j]*np.sign(a)/n
            else:
                grad_a += alpha[j]*m[j]*np.abs(a**(m[j]-1))*np.sign(a)/n

        if np.linalg.norm(learning_rate*grad_a)/n <= tol:
            break

        a = a - learning_rate*grad_a
        b = b - learning_rate*grad_b

    return np.concatenate([[b], a])
