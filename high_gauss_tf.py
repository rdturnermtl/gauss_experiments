from __future__ import print_function
import numpy as np
from scipy.integrate import quad
from scipy.optimize import approx_fprime, minimize
import scipy.stats as ss
import tensorflow as tf
from tensorflow.contrib.distributions import Uniform, Chi2
from high_gauss import JS_3 as JS_np


def quadv(f, a, b, args=()):
    N = len(f(a, *args))

    v = np.zeros(N)
    for ii in xrange(N):
        f_sub = lambda x: f(x, *args)[ii]
        v[ii], _ = quad(f_sub, a, b)
    return v


a = tf.placeholder(tf.float32)
L = tf.placeholder(tf.float32)
k = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32)

P = Chi2(k)
Q = Uniform(low=a, high=a + L)

log_p = P.log_prob(x)
log_q = Q.log_prob(x)
delta = log_p - log_q

integrand_1 = tf.exp(log_p) * tf.log_sigmoid(delta)
integrand_2 = tf.exp(log_q) * tf.log_sigmoid(-delta)
dintegrand_1 = tf.gradients(integrand_1, [a, L])
dintegrand_2 = tf.gradients(integrand_2, [a, L])

sess = tf.Session()

def int1_f(x_, a_, L_, k_):
    v = sess.run(integrand_1, {a: a_, L: L_, k: k_, x: x_})
    return v

def int2_f(x_, a_, L_, k_):
    v = sess.run(integrand_2, {a: a_, L: L_, k: k_, x: x_})
    return v


def int1_grad(x_, a_, L_, k_):
    v = sess.run(dintegrand_1, {a: a_, L: L_, k: k_, x: x_})
    return v


def int2_grad(x_, a_, L_, k_):
    v = sess.run(dintegrand_2, {a: a_, L: L_, k: k_, x: x_})
    return v


def get_JS(theta, k_):
    a_, L_ = theta

    P1, _ = quad(int1_f, 0.0, np.inf, (a_, L_, k_))
    P2, _ = quad(int2_f, a_, a_ + L_, (a_, L_, k_))
    JS = np.log(2) + 0.5 * (P1 + P2)
    return JS


def get_JS_grad(theta, k_):
    a_, L_ = theta

    P1, _ = quad(int1_f, 0.0, np.inf, (a_, L_, k_))
    P2, _ = quad(int2_f, a_, a_ + L_, (a_, L_, k_))
    JS = np.log(2) + 0.5 * (P1 + P2)

    dP1 = quadv(int1_grad, 0.0, np.inf, (a_, L_, k_))
    dP2 = quadv(int2_grad, a_, a_ + L_, (a_, L_, k_))

    # Add correction to derivs due to vars being in integration limits
    #b = np.nextafter(np.float32(a_ + L_), np.float32(-np.inf))
    b = np.float32(a_ + L_)
    b = b - 10 * np.spacing(b)

    lower = int2_f(a_, a_, L_, k_)
    upper = int2_f(b, a_, L_, k_)
    dP2_a = (upper - lower) + dP2[0]
    dP2_L = upper + dP2[1]

    dJS = 0.5 * (dP1 + np.asarray([dP2_a, dP2_L]))
    return JS, dJS


#init = tf.global_variables_initializer()
#sess.run(init)


def JS_(theta, k_):
    a_, L_ = theta
    JS = JS_np(k_, a_, L_)
    return JS


def get_JS_log(log_theta, k_):
    JS = get_JS(np.exp(log_theta), k_)
    return JS


def get_JS_grad_log(log_theta, k_):
    theta = np.exp(log_theta)
    JS, dJS = get_JS_grad(theta, k_)
    dJS_log = dJS * theta  # Make derivs wrt log scale
    return JS, dJS_log


def JS_log(log_theta, k_):
    JS = JS_(np.exp(log_theta), k_)
    return JS

'''
for _ in xrange(100):
    print('----')
    kk = np.random.uniform(2, 5)
    aa = np.random.rand() * (kk ** 2)
    LL = 2 * np.random.rand() * (kk ** 2 - aa)
    #theta = np.log([aa, LL])
    theta = [aa, LL]

    print([get_JS(theta, kk), JS_np(kk, aa, LL)])
    print(get_JS_grad(theta, kk))
    print('>>>')
    for ee in np.logspace(-15, -5, 10):
        print(approx_fprime(theta, JS_, ee, (kk,)))
'''

import matplotlib.pyplot as plt

k_grid = np.logspace(1, 3, 20)
v = []
score = []
for kk in k_grid:
    a0, b0 = ss.chi2.ppf([0.05, 0.95], kk)
    x0 = [a0, b0 - a0]
    res = minimize(get_JS_grad, x0, args=(kk,), method='L-BFGS-B', jac=True,
                   bounds=[(0.0, None), (0.0, None)],
                   options={'disp': True})
    v.append(res.x)
    score.append(JS_(res.x, kk))

    aa, LL = res.x

    aa_grid = np.linspace(0.0, 2.0 * max(a0, aa), 50)
    JS_grid = np.zeros(50)
    for ii in xrange(len(aa_grid)):
        theta = [aa_grid[ii], LL]
        JS_grid[ii] = JS_(theta, kk)
    plt.figure()
    plt.plot(aa_grid, JS_grid)
    plt.plot([aa, aa], [min(JS_grid), max(JS_grid)], 'k')
    plt.show()

    LL_grid = np.linspace(0.0, 2.0 * max(b0 - a0, LL), 50)
    JS_grid = np.zeros(50)
    for ii in xrange(len(LL_grid)):
        theta = [aa, LL_grid[ii]]
        JS_grid[ii] = JS_(theta, kk)
    plt.figure()
    plt.plot(LL_grid, JS_grid)
    plt.plot([LL, LL], [min(JS_grid), max(JS_grid)], 'k')
    plt.show()

print(zip(v, score))
