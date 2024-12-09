import Poly_Regression as pr
import numpy as np
import numpy.polynomial.polynomial as p
import pandas as pd
import json

l = -1
r = 1
n = 25
d = 5
m = 4
c = []
c_mon = []
c_cheb = []
diff = []
x = []
y = []
res1 = []
res2 = []
data = []
x0 = []

for i in range(10000):
    a = np.random.randint(-3, 4, d+1)
    x = np.sort(np.random.uniform(l, r, n))
    y = p.polyval(x, a) + np.random.normal(0, 1, n)*0.20

    model = pr.PolyLeastSquares(degree=d, basis='monomial', learning_rate=1e-3,
                                 max_iterations=10000, tol=0.0, normalize=True)
    model.fit(x, y, method='gd')
    r1 = model.predict(x)
    res1.append(np.linalg.norm(r1 - y)**2)
    c_mon.append(model.get_coefficients())

    model.normalize = False
    model.basis = 'chebyshev'
    model.fit(x, y, method='gd')
    r2 = model.predict(x)
    res2.append(np.linalg.norm(r2 - y)**2)
    c_cheb.append(model.get_coefficients())

    x0.append(x.tolist())
    data.append(y.tolist())
    c.append(a)


res1 = [r.tolist() for r in res1]
res2 = [r.tolist() for r in res2]
c = [r.tolist() for r in c]
c_mon = [r.tolist() for r in c_mon]
c_cheb = [r.tolist() for r in c_cheb]

pd.DataFrame({'normalized monomial': res1,
            'chebyshev': res2,
            'origional coefficients':c,
            }).to_csv('residuals.csv')

with open('residuals.json', 'w') as f:
    json.dump({'normalized monomial': res1,
               'chebyshev': res2,
               'origional coefficients':c,
               'monomial coefficients': c_mon,
               'chebyshev coefficients': c_cheb,
                'data': data,
                'x': x0
               }, f)

