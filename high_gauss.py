from __future__ import print_function
import numpy as np
from scipy.integrate import quad
from scipy.misc import logsumexp
from scipy.stats import chi, chi2, uniform, powerlaw
from scipy.special import expit as logistic
import scipy.stats as ss

base_dist = chi


def mult0(a, b):
    if a == 0:
        return a
    return a * b


def log_logistic(x):
    v = np.minimum(x, 0.0) - np.log(1 + np.exp(-np.abs(x)))
    return v


def mix_logpdf(x, k, a, L):
    lp1 = base_dist.logpdf(x, df=k)
    lp2 = powerlaw.logpdf(x, a=k, loc=a, scale=L)

    logpdf = logsumexp([lp1, lp2]) + np.log(0.5)

    #logpdf_chk = np.log(0.5 * np.exp(lp1) + 0.5 * np.exp(lp2))
    #print logpdf, logpdf_chk
    #assert(np.allclose(logpdf, logpdf_chk))

    return logpdf


def KL_chi_mix(k, a, L):
    def integrand(x):
        v = base_dist.pdf(x, k) * (base_dist.logpdf(x, df=k) - mix_logpdf(x, k, a, L))
        return v
    KL, err = quad(integrand, 0.0, np.inf)
    return KL


def KL_unif_mix(k, a, L):
    def integrand(x):
        v = powerlaw.pdf(x, a=k, loc=a, scale=L) * (powerlaw.logpdf(x, a=k, loc=a, scale=L) - mix_logpdf(x, k, a, L))
        return v
    KL, err = quad(integrand, a, a + L)
    return KL


def JS(k, a, L):
    JS = 0.5 * KL_chi_mix(k, a, L) + 0.5 * KL_unif_mix(k, a, L)
    return JS


def JS_2(k, a, L):
    def integrand(x):
        log_p = base_dist.logpdf(x, df=k)
        log_q = powerlaw.logpdf(x, a=k, loc=a, scale=L)
        delta = log_p - log_q
        v = mult0(np.exp(log_p), log_logistic(delta)) + \
            mult0(np.exp(log_q), log_logistic(-delta))
        return v
    JS_, _ = quad(integrand, 0.0, np.inf)
    JS = np.log(2) + 0.5 * JS_
    return JS


def JS_3(k, a, L):
    def integrand_1(x):
        log_p = base_dist.logpdf(x, df=k)
        log_q = powerlaw.logpdf(x, a=k, loc=a, scale=L)
        delta = log_p - log_q
        v = np.exp(log_p) * log_logistic(delta)
        return v

    def integrand_2(x):
        log_p = base_dist.logpdf(x, df=k)
        log_q = powerlaw.logpdf(x, a=k, loc=a, scale=L)
        delta = log_p - log_q
        v = np.exp(log_q) * log_logistic(-delta)
        return v

    P1, _ = quad(integrand_1, 0.0, np.inf)
    P2, _ = quad(integrand_2, a, a + L)
    JS = np.log(2) + 0.5 * (P1 + P2)
    return JS


def JS_(theta, k_):
    a_, L_ = theta
    JS = JS_3(k_, a_, L_)
    print(['dbg', theta, JS])
    return JS

if __name__ == '__main__':
    print(JS(2, 3, 2.5))
    print(JS_2(2, 3, 2.5))
    print(JS_3(2, 3, 2.5))

'''
    from skopt import gp_minimize
    k_grid = np.logspace(1, 3, 20)
    v = []
    score = []
    for kk in k_grid:
        med, w = ss.chi2.ppf([0.5, 0.9], kk)

        f = lambda x: JS_(x, kk)
        res = gp_minimize(f, [(1e-6, med), (1e-6, w)], noise=1e-6, verbose=True)
        print([kk, '>>>>', res.x, f(res.x)])
        v.append(res.x)
        score.append(JS_(res.x, kk))
    print(zip(v, score))
'''
