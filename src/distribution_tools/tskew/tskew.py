import numpy as np
import scipy.interpolate
from numbers import Number

np.seterr(all='raise')

from .numbaSpecialFns import numba_gammaln as gammaln
from .numbaSpecialFns import numba_hyp2f1 as hyp2f1
from .numbaSpecialFns import numba_betainc as betainc


# from   scipy.special import gammaln, hyp2f1, hyp1f1
import matplotlib.pyplot as plt; plt.ion()
from scipy.integrate import quad
from scipy.stats import t as scipy_trv
from numba import cfunc
from scipy.optimize import minimize
import warnings
import scipy.io as sio
import scipy
from .root_finding import newton, brentq



plt.close('all')
from numba import vectorize, njit
import numba as nb


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
    C = (dim/2.) * np.log(df * np.pi)
    D = 0.5 * logdet
    E = -t * np.log(1 + (1./df) * maha)


    w = np.sqrt(scale)
    J = dev * skew / w

    # Old definition -- this works
    # J = dev * skew
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

        return -np.mean(llvals)

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

def ts_invcdf_opt(q, loc, scale, df, skew):

    def ffun(x):
        return tscdf(x, loc, scale, df, skew) - q

    def fprime(x):
        return tspdf_1d(x, loc, scale, df, skew)

    # Do a single Newton iteration
    p0 = 0.0
    fval = ffun(p0)
    fder = fprime(p0)
    newton_step = fval / fder


    # Newton step
    p = p0 - newton_step

    while ffun(p) * fval > 0:
        p = p - newton_step

    if p0 < p:
        r = brentq(ffun, p0, p)
    else:
        r = brentq(ffun, p, p0)

    xvals = np.linspace(-20, 20, 1_000)
    fvals = ffun(xvals)
    # r = newton(func = f, x0 = 0.0, fprime = fprime)

    return r


def ts_invcdf(quantiles, loc, scale, df, skew):
    try:
        roots = np.zeros_like(quantiles)
        for count, q in enumerate(quantiles):
            r = ts_invcdf_opt(q, loc, scale, df, skew)
            pass

    except:
        pass

    pass


def numerical_inverse(rv_domain, cdf_vals):
    return scipy.interpolate.interp1d(cdf_vals, rv_domain, kind='cubic', fill_value="extrapolate")
    # def inv_fn(x):
    #     # Want inputs to be numbers between zero and one
    #     # sampled_data = np.interp(x, cdf_vals, rv_domain)
    #
    #     # return sampled_data
    #
    # return inv_fn

def tskew_moments(loc, scale, df, skew):
    w = np.sqrt(scale)
    alpha = skew

    omega = scale
    omega_bar = scale / (w * w)

    delta = (alpha * omega_bar) / np.sqrt(1 + alpha * omega_bar * alpha)

    gamma_div = np.exp(gammaln(0.5 * (df -1)) - gammaln(0.5 * df))
    mu = delta * np.sqrt(df / np.pi) * gamma_div

    expected_value_zero_loc = omega * mu
    expected_value = expected_value_zero_loc + loc

    second_moment = w**2 * (df / (df - 2))

    variance = second_moment - expected_value_zero_loc**2


    skew_f1 = mu
    skew_f2 = (df * (3 - delta**2) / (df - 3) - 3*df/(df - 2) + 2*mu**2)
    skew_f3 = np.power(df/(df - 2) - mu**2, -3/2)

    skewness = skew_f1 * skew_f2 * skew_f3



    kurt_f1_s1 = 3 * df**2 / ((df - 2) * (df-4))
    kurt_f1_s2 = -(4 * mu**2 * df * (3 - delta**2) / (df - 3))
    kurt_f1_s3 = 6 * mu**2 * df/(df -2)
    kurt_f1_s4 = -3*mu**4
    kurt_f1 = kurt_f1_s1 + kurt_f1_s2 + kurt_f1_s3 + kurt_f1_s4

    kurt_f2_s1 = df/(df - 2)
    kurt_f2_s2 = -mu**2
    kurt_f2 = np.power(kurt_f2_s1 + kurt_f2_s2, -2)

    kurtosis = kurt_f1*kurt_f2 - 3


    return expected_value, variance, skewness, kurtosis




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

