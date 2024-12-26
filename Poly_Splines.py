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
    def __init__(self, x, y, spline_degree, points_per_interval, basis='monomial'):
        # inputs:
        # x: array of points where data was collected
        # y: array of observed values at points x
        # spline_degree: degree of polynomials to use in each interval
        # points_per_interval: number of points to use in each interval (inclusive of endpoints)
        self.x = x
        self.y = y
        self.spline_degree = spline_degree
        self.points_per_interval = points_per_interval
        self.n = len(x) - 1
        self.m = self.n * spline_degree


    def calc_Splines(self):
        
        return
    

    def plot_data(self):
        plt.plot(self.x, self.y, 'ro')
        plt.show()
        return 
    

    def plot_splines(self):
        return

    
    def plot_spline(self, i):
    # plot the ith spline (blue dots represent the endpoints of the interval)
        return

    
    def plot_fit(self):
    # plot the piecewise polynomial fit together with the data points
        return