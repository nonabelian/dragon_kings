'''
    Author: Dylan Albrecht
    Date: December 13th, 2016

    This script fits a generic log-periodic function as from (e.g.):

    https://arxiv.org/abs/cond-mat/0201458v1

    * Uses scipy curve_fit to attempt a nonlinear least squares fit

    * As a demonstration, fits fake data in the main block.

    * Be mindful of parameter ordering:
      o, m, A, B, C, tau
'''
import os

import numpy as np
import scipy.stats as scs
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


# Log periodic function
def y(x, o, m, A, B, C, tau):
    return A - B * tau**m * x**m + C * tau**m * x**m * np.cos(o * np.log(x))


def res_fun(params, x, y_fit):
    return y(x, params[0], params[1], params[2], params[3], params[4],
             params[5]) - y_fit


# Log periodic function #2
def y_2(x, o, m, tau):
    ''' Function of the nonlinear parameters o, m, tau only.  Solves
        for the optimal 'slave' parameters X_hat(o, m, tau)
        OUTPUT: Function value
    '''
    A_hat, B_hat, C_1_hat, C_2_hat = minABC_y(xd, o, m, tau)

    return A_hat \
           + B_hat * (tau - x)**m \
           + C_1_hat * (tau - x)**m * np.cos(o * np.log(tau - x)) \
           + C_2_hat * (tau - x)**m * np.sin(o * np.log(tau - x))

def res_fun_2(params, x, y_fit):
    ''' Residual function -- required for the minimization problem
        OUTPUT: Residual
    '''
    return y_2(x, params[0], params[1], params[2]) - y_fit


def minABC_y(x, o, m, tau):
    ''' Solves for the 'slave' variables A, B, C_1, C_2
        INPUT: Nonlinear parameters o, m, tau
        OUTPUT: Returns linear parameter solutions (A, B, C_1, C_2)
    '''
    yi = yd
    fi = (tau - x)**m
    gi = (tau - x)**m * np.cos(o * np.log(tau - x))
    hi = (tau - x)**m * np.sin(o * np.log(tau - x))
    N = float(len(x))

    mat = np.array([[N, np.sum(fi), np.sum(gi), np.sum(hi)],
                    [np.sum(fi), fi.dot(fi), fi.dot(gi), fi.dot(hi)],
                    [np.sum(gi), fi.dot(gi), gi.dot(gi), gi.dot(hi)],
                    [np.sum(hi), fi.dot(hi), gi.dot(hi), hi.dot(hi)]])

    s_vec = np.array([np.sum(yi),
                      yi.dot(fi),
                      yi.dot(gi),
                      yi.dot(hi)]).reshape(-1, 1)
    vec_hat = np.linalg.inv(mat).dot(s_vec)

    return tuple(vec_hat.flatten())



if __name__ == '__main__':
    # Fit parameters
    o = 20.0
    m = 2.0
    A = 1.0
    B = -2.0
    C = 0.5
    t = 3.0

    # Fake data -- log-periodic function with Gaussian noise:
    num_data_points = 1000
    xd = np.linspace(0.1, 10, num_data_points)

    noise = 20
    yd = y(xd, o, m, A, B, C, t) + scs.norm(0, noise).rvs(size=num_data_points)

    ################
    # Curve Fitting

    p0 = (21.0, m, A, B, C, t)
    popt, pcov = curve_fit(y, xd, yd, p0=np.ones(6))

    st = "o: {0}, m: {1}, A: {2}, B: {3}, C: {4}, tau: {5}"
    params = "omega: {0}, m: {1}".format(o, m)
    print "Specified parameters"
    print '-' * 30
    print st.format(o, m, A, B, C, t)
    print '-' * 30
    print "Fit parameters"
    print '-' * 30
    print popt
    print '-' * 30

    plt.scatter(xd, yd)
    plt.plot(xd, y(xd, *popt), c='r')
#    plt.plot(x, y(x, o, m, A, B, C, t))

    fname = "lppl_curve_fit_fit.png"
    save_file = os.path.join('images', fname)

    plt.savefig(save_file)
    plt.show()
