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


def b_splines(deg, knots, interior):
    # deg: degree of the polynomials that form the splines
    # knots: array of knots
    # interior: 1x2 array, [a, b], where a is the lowest index of an 
    #           interior knot and b is the highest index of an interior knot
    #           N.B. the number of exterior knots must be 2*deg
    # returns: list of polynomials that define the B-splines
    
    n = interior[1] - interior[0] + deg
    splines = []
    
    for i in range(n):
        splines.append(spline(knots[i:i+deg+2]))

    return splines


def spline(t):
    # t: the set of points used to find the spline
    #    degree is not specified, it's infered from the length of t
    #    i.e. the degree of the spline is len(t) - 1
    L=[]

    for i in range(len(t)-1): # loop through the splines of order i
        L.append([])
        for j in range(len(t)-i-1): # loop through the knots
            if i==0:
                L[i].append([1])
            else:
                L[i].append([])
                P1 = [p.polymul([-t[j], 1], L[i-1][j][k])/(t[j+i] - t[j]) for k in range(i)]
                P2 = [p.polymul([t[j+i+1], -1], L[i-1][j+1][k])/(t[j+i+1] - t[j+1]) for k in range(i)]

                L[i][j].append(P1[0])
                if i > 1:
                    for k in range(i-1):
                        L[i][j].append(p.polyadd(P1[1+k], P2[k]))
                L[i][j].append(P2[-1])

    return L[-1][0]
