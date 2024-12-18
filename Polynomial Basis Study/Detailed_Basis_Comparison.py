import Poly_Regression as pr
import numpy as np
import numpy.polynomial.polynomial as p
import os
import json


os.makedirs('./Basis_Compare_Results', exist_ok=True)

# degree of polynomial used to generate synthetic data
for dd in range(2, 8):

    res_rm = []
    res_rc = []
    orig_coeff = []
    mon_coeff = []
    che_coeff = []
    num_points = []
    fit_deg = []

    for dm in range(2, 8):
        monomial_model = pr.PolyLeastSquares(degree=dm, basis='monomial', learning_rate=1e-3,
                                    max_iterations=10000, tol=0.0, normalize=True)
        
        chebyshev_model = pr.PolyLeastSquares(degree=dm, basis='chebyshev', learning_rate=1e-3,
                                    max_iterations=10000, tol=0.0)

        # number of points used to generate synthetic data
        for n in range(25, 101, 25):
            xe = np.linspace(-1, 1, n)

            for i in range(100):
                a = np.random.randint(-30, 40, dd+1)/10

                # fit models on randomly sampled synthetic data
                xr = np.sort(np.random.uniform(-1, 1, n))
                yr = p.polyval(xr, a) + np.random.normal(0, 1, n)*0.20

                monomial_model.fit(xr, yr, method='gd')
                chebyshev_model.fit(xr, yr, method='gd')

                res_rm.append(np.linalg.norm(monomial_model.predict(xe) - p.polyval(xe, a))**2)
                res_rc.append(np.linalg.norm(chebyshev_model.predict(xe) - p.polyval(xe, a))**2)
                orig_coeff.append(a.tolist())
                mon_coeff.append(monomial_model.get_coefficients())
                che_coeff.append(chebyshev_model.get_coefficients())
                num_points.append(n)
                fit_deg.append(dm)

    with open('./Basis_Compare_Results/data_degree_'+str(dd)+'.json', 'w') as f:
        json.dump({'monomial residuals': res_rm
                    ,'chebyshev residuals': res_rc
                    ,'original_coefficients': orig_coeff
                    ,'monomial_coefficients': mon_coeff
                    ,'chebyshev_coefficients': che_coeff
                    ,'number_points':num_points
                    ,'fit_degree': fit_deg
                    }, f)
    print('Finished degree: ', dd)

    
