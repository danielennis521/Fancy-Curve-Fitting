import matplotlib.pyplot as plt
import Poly_Regression as pr
import numpy as np
import polynomial_basis as pb
import numpy.polynomial.polynomial as p
import pandas as pd

l = -1
r = 1
n = 25
d = 5
m = 4
c = []
diff = []
x = []
y = []
res1 = []
res2 = []
z = np.linspace(l, r, 1000)

for i in range(10000):
    a = np.random.randint(-3, 3, d+1)
    x = np.random.uniform(l, r, n)
    y = np.polyval(a, x) + np.random.normal(0, 1, n)*0.20

    model = pr.PolyLeastSquares(degree=d, basis='monomial', learning_rate=1e-3,
                                 max_iterations=10000, tol=0.0, normalize=True)
    model.fit(x, y, method='gd')
    r1 = model.predict(x)
    res1.append(np.linalg.norm(r1 - y)**2)

    model.normalize = False
    model.basis = 'chebyshev'
    model.fit(x, y, method='gd')
    r2 = model.predict(x)
    res2.append(np.linalg.norm(r2 - y)**2)

    c.append(a)

pd.DataFrame({'normalized monomial': res1, 'chebyshev': res2, 'coefficients':c}).to_csv('residuals.csv')

