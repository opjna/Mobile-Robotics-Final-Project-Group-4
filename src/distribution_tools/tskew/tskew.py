import math

import numpy as np
import scipy.interpolate
from numbers import Number

np.seterr(all='raise')
# from   scipy.special import gammaln, hyp2f1, hyp1f1
import matplotlib.pyplot as plt; plt.ion()
from scipy.integrate import quad
from scipy.stats import t as scipy_trv
from numba import cfunc
from functools import partial
import timeit
from scipy.optimize import minimize
import warnings
import scipy.io as sio
import scipy
# from root_finding import newton, brentq

plt.close('all')
from numba import vectorize, njit
import numba as nb


from .numbaSpecialFns import numba_gammaln as gammaln
from .numbaSpecialFns import numba_hyp2f1 as hyp2f1
from .numbaSpecialFns import numba_betainc as betainc

# Source: https://gregorygundersen.com/blog/2020/01/20/multivariate-t/
# @article{kollo2021multivariate,
#   title={Multivariate Skew t-Distribution: Asymptotics for Parameter Estimators and Extension to Skew t-Copula},
#   author={Kollo, T{\~o}nu and K{\"a}{\"a}rik, Meelis and Selart, Anne},
#   journal={Symmetry},
#   volume={13},
#   number={6},
#   pages={1059},
#   year={2021},
#   publisher={Multidisciplinary Digital Publishing Institute}
# }

@njit
def tcdf_1d(x, df):
    # if use_scipy:
    #     # Scipy-computed tcdf vals
    #     scipy_tcdf_vals = scipy_trv.cdf(x, df)
    #     return scipy_tcdf_vals
    # else:
    # A = gammaln((df+1.)/2.)
    # B = hyp2f1(0.5, (df+1.)/2., 1.5, -x**2 / df)
    #
    #
    # # BA = hyp1f1(0.5, 1.5, -x**2)
    # C = np.sqrt(np.pi * df)
    # D = gammaln(df/2.)
    #
    # tcdf_vals = 0.5 + x * B * np.exp(A - D) / C


    tcdf_vals_v2 = 0.5 * betainc(0.5 * df, 0.5, df / (df + x**2))
    inds = x > 0
    tcdf_vals_v2[inds] = 1 - tcdf_vals_v2[inds]
    return tcdf_vals_v2




def tspdf_1d_scipy(x, loc, scale, df, skew):
    z = (x - loc)/np.sqrt(scale)
    return 2 * scipy_trv.pdf(z, df + 1) * scipy_trv.cdf(skew * z, df + 1)

@njit
def tspdf_1d(x, loc, scale, df, skew):
    return np.exp(tslogpdf_1d(x, loc, scale, df, skew))


@njit
def tslogpdf_1d(x, loc, scale, df, skew):
    dim = 1
    vals, vecs = scale, np.array([1])

    logdet     = np.log(scale)
    valsinv    = np.array([1./vals])
    U          = vecs * np.sqrt(valsinv)
    dev        = x - loc
    maha       = np.square(dev * U)


    t = 0.5 * (df + dim)
    A = gammaln(t)
    B = gammaln(0.5 * df)
    C = dim/2. * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -t * np.log(1 + (1./df) * maha)


    J = dev * skew
    rad = np.sqrt((dim + df) / (maha + df))

    Fval = tcdf_1d(J * rad, dim + df)

    F = np.log(2) + np.log(Fval)

    return A - B - C - D + E + F


def sampleCvM(dataSort, CDF):
    N = len(dataSort)
    CDF_vals = CDF(dataSort)
    empirical_CDF = np.linspace(1, 2*N - 1, 2) / (2 * N)

    diff_CDFs = CDF_vals - empirical_CDF
    CvM = 1/(12 * N) + np.sum(diff_CDFs ** 2)


def getIntegrand(loc, scale, df, skew):

    @njit(cache=True)
    def integrand(y):
        return tspdf_1d(y, loc, scale, df, skew)[0]

    return integrand


# def tscdf(x, loc, scale, df, skew):
#     tscdf_vals = np.zeros_like(x)
#
#     integrand = getIntegrand(loc, scale, df, skew)
#     nb_integrand = cfunc("float64(float64)")(integrand)
#     for index, upper_limit in enumerate(x):
#         # integral_val, abs_err = quad(lambda y: tspdf_1d(y, loc, scale, df, skew), -np.inf, upper_limit, epsrel = 1e-4)
#         integral_val, abs_err = quad(nb_integrand.ctypes, -np.inf, upper_limit)
#
#         tscdf_vals[index] = integral_val
#
#
#     return tscdf_vals



def tscdf(x, loc, scale, df, skew):
    # tscdf_vals = np.zeros_like(x)

    if isinstance(x, Number):
        x = np.array([x])
    integrand = getIntegrand(loc, scale, df, skew) # Closure that captures the parameters of the distribution
    nb_integrand = cfunc("float64(float64)")(integrand) # Convert it cfunc for faster integration

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if len(x) == 1:
            tscdf_vals = quad(nb_integrand.ctypes, -np.inf, x)[0]
        else:
            partial_integrals = np.zeros_like(x)
            partial_integrals[0] = quad(nb_integrand.ctypes, -np.inf, x[0])[0]

            sliding_windows = np.lib.stride_tricks.sliding_window_view(x, 2)

            for index, window in enumerate(sliding_windows):
                integral_val, abs_err = quad(nb_integrand.ctypes, window[0], window[1])

                partial_integrals[index + 1] = integral_val

            tscdf_vals = np.cumsum(partial_integrals)

    # for index, upper_limit in enumerate(x):
    #     # integral_val, abs_err = quad(lambda y: tspdf_1d(y, loc, scale, df, skew), -np.inf, upper_limit, epsrel = 1e-4)
    #     integral_val, abs_err = quad(nb_integrand.ctypes, -np.inf, upper_limit)
    #
    #     tscdf_vals[index] = integral_val


    return tscdf_vals

def getObjectiveFunction(data, use_loglikelihood = False):

    sorted_data = np.sort(data)
    N = len(data)


    def objFun(theta):
        loc = theta[0]
        scale = theta[1]
        df = theta[2]
        skew = theta[3]

        tscdf_vals = tscdf(sorted_data, loc, scale, df, skew)

        empirical_CDF = np.arange(1, 2 * N , 2) / (2 * N)

        diff_CDFs = tscdf_vals - empirical_CDF
        CvM = 1 / (12 * N) + np.sum(diff_CDFs ** 2)
        return CvM

    @njit
    def loglikelihood(theta):
        loc = theta[0]
        scale = theta[1]
        df = theta[2]
        skew = theta[3]

        llvals = tslogpdf_1d(sorted_data, loc, scale, df, skew)

        return -np.sum(llvals)

    if use_loglikelihood:
        return loglikelihood
    else:
        return objFun


@njit
def root_Newton_Rhapson(fun, x0, jac, tol=1e-12, maxiter=100):
    for _ in range(maxiter):
        fval = fun(x0)

        if np.abs(fval) < tol:
            x1 = x0
            break
        fder = jac(x0)

        newton_step = fval / fder
        x1 = x0 - newton_step

        if np.abs(x1) < tol:
            break

        x0 = x1

    return x1

def ts_invcdf(q, loc, scale, df, skew):

    def f(x):
        return tscdf(x, loc, scale, df, skew) - q

    def fprime(x):
        return tspdf_1d(x, loc, scale, df, skew)
    # Do a single Newton iteration
    p0 = 0.0
    fval = f(p0)
    fder = fprime(p0)
    newton_step = fval / fder
    # Newton step
    p = p0 - newton_step

    if p0 < p:
        r = brentq(f, p0, p)
    else:
        r = brentq(f, p, p0)

    xvals = np.linspace(-20, 20, 1_000)
    fvals = f(xvals)
    # r = newton(func = f, x0 = 0.0, fprime = fprime)

    return r

def numerical_inverse(rv_domain, cdf_vals):
    return scipy.interpolate.interp1d(cdf_vals, rv_domain, kind='cubic', fill_value="extrapolate")
    # def inv_fn(x):
    #     # Want inputs to be numbers between zero and one
    #     # sampled_data = np.interp(x, cdf_vals, rv_domain)
    #
    #     # return sampled_data
    #
    # return inv_fn




if __name__ == '__main__':
    x = np.linspace(-6, 6, 1_000)
    # res = tcdf_1d(x, 3)
    # fig, axs = plt.subplots(1, 3, sharex=True)
    # for df in [1, 2, 5, 100]:
    #     p = tspdf_1d(x, 0, 1, df, skew = 0)
    #     axs[0].plot(x, p)
    #
    # for s in [1, 2, 5, 10]:
    #     p = tspdf_1d(x, 0, s, 2, xi = 0)
    #     axs[1].plot(x, p)
    #
    # for xi in [0, -1, -2, -5, -5000]:
    #     p = tspdf_1d(x, 0, 1, 2, xi)
    #     axs[2].plot(x, p)
    #
    # p_ev = tspdf_1d(x, 0.5, 2, 3, 5)


    # fig, axs = plt.subplots(1, 2)
    # axs[0].plot(p_ev, label = 'Erick')


    # ef_integral = quad(lambda x: tspdf_1d(x, 0.5, 2, 3, 5), -np.inf, np.inf)

    loc = 1
    scale = 2
    df = 5
    skew = 1

    q = 0.5
    tslogpdf_1d(q, loc, scale, df, skew)

    N = 1_000
    xvals = np.linspace(-10, 10, N)
    cdf = tscdf(xvals, loc, scale, df, skew )
    pdf = tspdf_1d(xvals, loc, scale, df, skew)

    invcdf = numerical_inverse(xvals, cdf)

    uniform_realization = np.random.rand(N)
    skewt_realization = invcdf(uniform_realization)
    # logpdf = tslogpdf_1d(x, loc, scale, df, skew)

    # t = timeit.Timer(partial(tscdf, x, loc, scale, df, skew))
    # print(t.timeit(1))
    cdf = tscdf(x+2, loc, scale, df, skew)

    # plt.figure()
    # plt.plot(x, cdf)
    theta = np.array([loc + 0.2, scale - 1, df+2, skew-2])

    test_data = sio.loadmat('/home/efvega/data/copulaTesting/2d_data.mat')['data'][1, :]
    obj_fun = getObjectiveFunction(skewt_realization, use_loglikelihood=False)
    obj_fun(theta)

    def callbackF(theta):
        print(theta)

    res = minimize(getObjectiveFunction(skewt_realization, use_loglikelihood=True), x0 = theta, callback=callbackF, method='Nelder-Mead')
    print('Now switching')
    # res = minimize(getObjectiveFunction(skewt_realization, use_loglikelihood=False), x0 = res.x, callback=callbackF, method='Nelder-Mead')


    est_pdf = tspdf_1d(xvals, res.x[0], res.x[1], res.x[2], res.x[3])
    plt.hist(skewt_realization, bins = 50, density=True)
    plt.plot(xvals, est_pdf, linestyle = '--')
    plt.plot(xvals, pdf)

    # print(res_initial.x)
    print(res.x)



    # integrand = getIntegrand(loc, scale, df, skew)
    # nb_integrand = cfunc("float64(float64)")(integrand)

