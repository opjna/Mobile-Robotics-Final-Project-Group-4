from scipy.stats import norm, expon, gamma, weibull_min, rayleigh, beta, binom, poisson, geom
import matplotlib.pyplot as plt; plt.ion()

from scipy.optimize import minimize
from tskew.tskew import getObjectiveFunction, tspdf_1d

import numpy as np

def gen_data_poly(N = 1000):
    # Normal
    nmean=1
    nloc=2
    normal_data = norm.rvs(loc=1, scale=2, size=N)
    normal_dist = norm(loc=1, scale=2)

    # Exponential
    lam = 2
    exponential_data = expon.rvs(scale = 1/lam, size = N)
    exponential_dist = expon(scale=1/lam)

    # Gamma
    a = 1
    b = 2
    gamma_data = gamma.rvs(a, scale=2, size = N)
    gamma_dist = gamma(a, scale=2)
    # Weibull
    # Rayleigh
    # Beta
    # Binomial
    # Poisson
    # Geometric
    # Negative Binomial

    data_dict = {
        'Normal': ([nmean, nloc], normal_data, normal_dist),
        'Exponential': ([lam], exponential_data, exponential_dist),
        'Gamma': ([a,b], gamma_data, gamma_dist)
    }
    return data_dict

if __name__ == "__main__":
    data = gen_data_poly()


    for dist_name, dist_data in data.items():
        realization = dist_data[1]
        distribution = dist_data[2]

        loc = np.mean(realization)
        scale = np.std(realization)
        df = 3
        skew = 1

        theta = np.array([loc, scale, df, skew])

        res = minimize(getObjectiveFunction(realization, use_loglikelihood=True), x0=theta,
                       method='Nelder-Mead')

        N = 1_000
        extent =  np.max(realization) - np.min(realization)
        xvals = np.linspace(np.min(realization) - 0.1 * extent, np.max(realization) + 0.1 * extent, N)

        plt.figure()
        est_pdf = tspdf_1d(xvals, res.x[0], res.x[1], res.x[2], res.x[3])
        plt.hist(realization, bins=50, density=True)
        plt.plot(xvals, distribution.pdf(xvals), label=f'True: {dist_name}', linewidth=5)
        plt.plot(xvals, est_pdf, linestyle='--', label='Estimated skew t', linewidth=5)
        plt.title(f'{dist_name}-distributed data and estimated skew ' + r'$t$-distribution')
        plt.legend()

        pass