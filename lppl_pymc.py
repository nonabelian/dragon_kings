'''
    Author: Dylan Albrecht
    Date: December 13th, 2016

    This script fits a generic log-periodic function as from (e.g.):

    https://arxiv.org/abs/cond-mat/0201458v1

    * Uses pyMC MCMC sampling to converge on a solution, providing parameter
      statistics, in addition.
    * As a demonstration, fits fake data in the main block.

    * Sensitive to prior distribution and hyperparameters for:
      o, m, A, C, tau
'''
import os

import pymc3 as pm
import numpy as np
import scipy.stats as scs
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt

from lppl_basinhopping import MyStepper
from lppl_basinhopping import MyBounds


# Log periodic function, as per https://arxiv.org/abs/cond-mat/0201458v1
def y(x, o, m, A, C, tau):
    ''' Target Log-Periodic Power Law (LPPL) function
        Note: Scaled 'B' -> -1.0
        TODO: Perhaps we should check that parameters passed result in
              reasonable computations
    '''
    ret = A + tau**m * x**m + C * tau**m * x**m * np.cos(o * np.log(x))

    return ret


if __name__ == '__main__':
    # Fake data parameters -- what we hope our fit returns!
    od = 20.0    # Frequency
    md = 2.0     # Power
    Ad = 2.0     # Intercept
    Cd = 0.5     # Coefficient
    td = 3.0     # Critical time

    # Fake data -- log-periodic function with Gaussian noise:
    num_data_points = 1000
    xd = np.linspace(0.1, 10, num_data_points)

    noise = 20
    yd = y(xd, od, md, Ad, Cd, td) \
         + scs.norm(0, noise).rvs(size=num_data_points)

    def E_func(params):
        ret = y(xd, params[0], params[1], params[2], params[3],
                params[4])

        n = float(len(ret))
        er = (ret - yd).dot(ret - yd) / n

        if np.isnan(er):
            er = 1e10

        return er

    p0 = [1.0, 1.0, 1.0, 1.0, 1.0]

    # Step size can greatly affect the solution/convergence.
    mystep = MyStepper(0.1)
    mybound = MyBounds()

    ret = basinhopping(E_func, p0, take_step=mystep, accept_test=mybound)

    o_b = ret['x'][0]
    m_b = ret['x'][1]
    A_b = ret['x'][2]
    C_b = ret['x'][3]
    t_b = ret['x'][4]

    #############
    # PyMC3 Model

    model = pm.Model()

    with model:
        o = pm.Normal('o', mu=o_b, sd=10)
        m = pm.Normal('m', mu=m_b, sd=10)
        A = pm.Normal('A', mu=A_b, sd=10)
        C = pm.Normal('C', mu=C_b, sd=10)
        t = pm.Normal('t', mu=t_b, sd=10)

        y_mu = A + t**m * xd**m + C * t**m * xd**m * np.cos(o * np.log(xd))
        y_sd = pm.HalfNormal('y_sd', sd=10)

        y_obs = pm.Normal('y_obs', mu=y_mu, sd=y_sd, observed=yd)

        trace = pm.sample(tune=1000)

    ppc = pm.sample_ppc(trace, model=model)
    y_model = ppc['y_obs']

    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111)
    for row in y_model:
        ax.scatter(xd, row, color='blue', alpha=0.01)

    ax.scatter(xd, yd, color='black', alpha=0.5)

    fname = "lppl_basinhopping_plus_pymc3_fit.png"
    save_file = os.path.join('images', fname)
    plt.savefig(save_file)

    plt.show()

