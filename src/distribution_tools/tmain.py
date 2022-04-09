from scipy.stats import norm, expon, gamma, weibull_min, rayleigh, beta, binom, poisson, geom
# import matplotlib
import matplotlib.pyplot as plt; plt.ion()
# matplotlib.use('TkAgg')

from scipy.optimize import minimize
from tskew.tskew import getObjectiveFunction, tspdf_1d, tskew_moments

import numpy as np

def gen_data_poly(N = 2000):
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
    x = 1

    # Rayleigh - TODO
    rayleigh_data = rayleigh.rvs(loc=nmean, scale=nloc, size = N)
    rayleigh_dist = rayleigh(loc=nmean, scale=nloc)
    # Beta - TODO
    beta_a=3
    beta_b=4
    beta_data = beta.rvs(a=beta_a, b=beta_b, scale = 2, size = N)
    beta_dist = beta(a=beta_a, b=beta_b, scale=2)
    # Binomial
    # n = 10
    # p = .1
    # binom_data = binom.rvs(n=n, p=p, loc=1, size=N)
    # binom_dist = binom(n=n, p=p, loc=1)
    # Poisson - TODO
    # mu_poisson = 2
    # poisson_data = poisson.rvs(mu=mu_poisson, size = N)
    # poisson_dist = poisson(mu=mu_poisson)
    # Geometric
    # Negative Binomial

    data_dict = {
        'Normal': ([nmean, nloc], normal_data, normal_dist),
        'Exponential': ([lam], exponential_data, exponential_dist),
        'Gamma': ([a,b], gamma_data, gamma_dist),
        'Rayleigh': ([nmean, nloc], rayleigh_data, rayleigh_dist),
        'Beta': ([beta_a, beta_b], beta_data, beta_dist),
        # 'Poisson': ([mu_poisson], poisson_data, poisson_dist),
        # 'Binomial': ([n, p], binom_data, binom_dist)
    }
    return data_dict

if __name__ == "__main__":
    data = gen_data_poly()


    for dist_name, dist_data in data.items():
        realization = dist_data[1]
        distribution = dist_data[2]

        loc = np.mean(realization)
        scale = np.var(realization)
        median = np.median(realization)
        df = 1000
        skew = (3 * loc - median) / np.sqrt(scale)

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
        plt.show()

        solution_params = res['x']
        loc_est, scale_est, df_est, skew_est = solution_params
        moments_from_est = tskew_moments(loc_est, scale_est, df_est, skew_est)
        pass
