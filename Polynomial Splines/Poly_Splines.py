# this file contains the class poly_Splines, which is used to create a piecewise polynomial function 
# that interpolates a set of data points. The motivation for implementing this as a class is to provide an
# easy way to work with the piecewise fit since it potentially involves a large number of polynomials, each
# with its own set of coefficients. 


import numpy as np
import numpy.polynomial.polynomial as p
from numpy import linalg as la
import matplotlib.pyplot as plt
import polynomial_basis as b


class B_Splines:
    def __init__(self, deg, knots, interior):
        # inputs:
        # deg: Degree of the polynomials used to build splines
        # knots: The whole set of knots used to form the splines
        # interior: 1x2 array, [a, b], where a is the lowest index of an 
        #           interior knot and b is the highest index of an interior knot
        #           N.B. the number of exterior knots must be 2*deg
        self.calc_splines(deg, knots, interior)
        self.weights = [1 for i in self.splines]
        print(self.weights)


    def calc_splines(self, deg, knots, interior):
        # inputs:
        # deg: Degree of the polynomials used to build splines
        # knots: The whole set of knots used to form the splines
        # interior: 1x2 array, [a, b], where a is the lowest index of an 
        #           interior knot and b is the highest index of an interior knot
        #           N.B. the number of exterior knots must be 2*deg
        self.splines = b.b_splines(deg, knots, interior)
        self.deg = deg
        self.knots = knots
        self.interior = interior


    def fit(self, X, Y):
        # inputs:
        # x: point where observation was made
        # y: observed value
        #   N.B. the measurements you want to fit need to lie within the
        #   set of interior knots

        self.x = X
        self.y = Y

        A = np.array([[self.eval_spline(x, j) for j in range(len(self.splines))] for x in X])
        self.weights = np.linalg.lstsq(A, Y, rcond=0)[0]
    

    def eval_spline(self, x, i):
        # inputs:
        # x: data point to evaluate
        # i: which spline to evaluate

        absolute_position = len(list(filter(lambda u: u<=x, self.knots))) - 1
        relative_position = absolute_position - i

        if relative_position >= 0 and relative_position < len(self.splines[i]):
            return p.polyval(x, self.splines[i][relative_position])
        else:
            return 0.0


    def predict(self, x):
        y = 0
        for i in range(len(self.weights)):
            y += self.weights[i] * self.eval_spline(x, i)
        return y
    

    def plot_spline_set(self, colors=['b', 'g', 'r', 'k']):
        for i in range(len(self.splines)):
            for j in range(len(self.splines[0])):
                t = np.linspace(self.knots[i+j], self.knots[i+j+1], 100)
                plt.plot(t, [p.polyval(z, self.splines[i][j]) for z in t], colors[i%len(colors)])
        plt.show()

    
    def plot_spline(self, i, color='b'):
    # plot the ith spline (blue dots represent the endpoints of the interval)
        for j in range(len(self.splines[i])):
            t = np.linspace(self.knots[i+j], self.knots[i+j+1], 100)
            plt.plot(t, [p.polyval(z, self.splines[i][j]) for z in t], color)
        plt.show()

    
    def plot_fit(self, data=False, fit_color='b', data_color='ro'):
    # plot the weighted sum of the splines
        for i in range(len(self.knots)-1):
            interval = np.linspace(self.knots[i], self.knots[i+1], 100)
            y = [self.predict(t) for t in interval]
            plt.plot(interval, y, fit_color)

        if data:
            plt.plot(self.x, self.y, data_color)
        plt.show()


    def plot_data(self):
        # plots the last set of data that was fit
        plt.plot(self.x, self.y, 'ro')
        plt.show()
