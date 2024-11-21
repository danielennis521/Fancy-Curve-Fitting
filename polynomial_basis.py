import numpy as np
import numpy.polynomial.polynomial as p


def rescale_polynomial(q, a0, b0, a1, b1):
    # Rescale the polynomial q from the interval [a0, b0] to [a1, b1]
    s = (b1 - a1) / (b0 - a0)
    new = [0]
    for i in range(len(q)):
        t = [1]
        for j in range(i):
            t = p.polymul(t, [a1 - a0*s, s])

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
    return


def Hermite_Polynomials(n):
    return
