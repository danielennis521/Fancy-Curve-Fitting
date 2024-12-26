import numpy as np
import numpy.polynomial.polynomial as p


def rescale_polynomial(q, a0, b0, a1, b1):
    # Rescale the polynomial q from the interval [a0, b0] to [a1, b1]
    s = (b0 - a0)/(b1 - a1) 
    new = [0]
    for i in range(len(q)):
        t = [1]
        for j in range(i):
            t = p.polymul(t, [a0 - a1*s, s])

        new = p.polyadd(new, p.polymul(q[i], t))

    return new


def monomials(n):
    return [ np.array(np.zeros(i).tolist() + [1.0]) for i in range(n)]


def Chebyshev_Polynomials(n, a=-1, b=1):
    for i in range(n):
        if i == 0:
            L = [np.array([1])]
        elif i == 1:
            L.append(np.array([0, 1]))
        else:
            L.append(p.polysub(p.polymul([0, 2], L[i-1]), L[i-2]))

    if a != -1 or b != 1:
        for i in range(n):
            L[i] = rescale_polynomial(L[i], -1, 1, a, b)

    return L


def Legendre_Polynomials(n, a=-1, b=1):
    # The Legendre polynomials are much more complicated to generate than the Chebyshev polynomials
    # because of this we simply store the first 10 to minimize how often we need to generate them.
    # For higher orders we will generate them on the fly using Rodrigues' formula.

    L = [np.array([1]),
         np.array([0, 1]),
         np.array([-0.5, 0, 1.5]),
         np.array([0, -1.5, 0, 2.5]),
         np.array([3, 0, -30, 0, 35])/8,
         np.array([0, 15, 0, -70, 0, 63])/8,
         np.array([-5, 0, 105, 0, -315, 0, 231])/16,
         np.array([0, -35, 0, 315, 0, -693, 0, 429])/16,
         np.array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435])/128,
         np.array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155])/128]
    
    if n <= 10:
        L = L[:n]
    else:
        for i in range(10, n):
            L.append(p.polyder(p.polypow([1, 0, -1] , i), m=i) / (2**i * np.math.factorial(i)))
    
    if a != -1 or b != 1:
        for i in range(n):
            L[i] = rescale_polynomial(L[i], -1, 1, a, b)
    
    return L
