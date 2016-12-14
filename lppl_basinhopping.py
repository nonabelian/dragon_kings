'''
    Author: Dylan Albrecht
    Date: December 13th, 2016

    This script fits a generic log-periodic function as from (e.g.):

    https://arxiv.org/abs/cond-mat/0201458v1

    * Uses scipy basinhopping, which is generally preferred over their
      simulated annealing algorithm.

    * As a demonstration, fits fake data in the main block.

    Reasonable steps are implemented in MyStepper
    Reasonable bounds are implemented in MyBounds

    * These are both sensitive to parameter ordering:
      o, m, A, C, tau
'''
import os

import numpy as np
import scipy.stats as scs
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt


# Log periodic function, as per https://arxiv.org/abs/cond-mat/0201458v1
def y(x, o, m, A, C, tau):
    ''' Target Log-Periodic Power Law (LPPL) function
        Note: Scaled 'B' -> -1.0
        TODO: Perhaps we should check that parameters passed result in
              reasonable computations
    '''
    ret = A + tau**m * x**m + C * tau**m * x**m * np.cos(o * np.log(x))

    return ret


class MyStepper(object):
    ''' Implements some simple reasonable modifications to parameter stepping

            * Frequencies are kept positive
            * Powers are exponentially suppressed from getting 'large'
            * 
    '''
    def __init__(self, stepsize=0.5):
        self.stepsize = stepsize

    def __call__(self, x):
        ''' Updater for stepping x[*] parameters
        '''

        s = self.stepsize

        # relative parameter scales o/m, etc.
        scale_om = 10.0
        scale_Am = 100.0
        scale_Cm = 100.0
        scale_tm = 10.0

        # Treat the frequency differently -- only interested in positive
        # frequency values.
        inc = scale_om * np.random.uniform(-s, s)

        if (x[0] + inc <= 0):
            x[0] += abs(inc)
        else:
            x[0] += inc

        # Treat the power differently -- large powers (>>1) can easily cause
        # numerical problems, so we exponentially penalize steps in the power
        # parameter 'm'.
        x[1] += np.exp(-abs(x[1])) * np.random.uniform(-s, s)

        x[2] += scale_Am * np.random.uniform(-s, s)
        x[3] += scale_Cm * np.random.uniform(-s, s)

        # Treat 'critical time' differently
        inc = scale_tm * np.random.uniform(-s, s)

        if (x[4] + inc <= 0):
            x[4] += abs(inc)
        else:
            x[4] += inc

        return x
        

class MyBounds(object):
    ''' This class implements a set of reasonable bounds (depends on units and
        domain knowledge):
            * frequencies less than 200
            * powers less than 10
            * intercept/coefficient less than 1e4
            * critical time
    '''
    def __init__(self, xmax=[200, 10, 1e4, 1e4, 100],
                       xmin=[0, -10, -1e4, -1e4, 1e-3]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        ''' Evaluater -- Returns True if we accept the solution, False
            otherwise.
        '''
        x = kwargs["x_new"]

        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))

        # Only accept solutions within all parameter bounds
        return tmax and tmin



if __name__ == '__main__':
    # Fake data parameters -- what we hope our fit returns!
    o = 20.0    # Frequency
    m = 2.0     # Power
    A = 2.0     # Intercept
    C = 0.5     # Coefficient
    t = 3.0     # Critical time

    panswer = [('o', 20.0), ('m', 2.0), ('A', 2.0), ('C', 0.5), ('tau', 3.00)]

    # Fake data -- log-periodic function with Gaussian noise:
    num_data_points = 1000
    xd = np.linspace(0.1, 10, num_data_points)

    noise = 20
    yd = y(xd, o, m, A, C, t) + scs.norm(0, noise).rvs(size=num_data_points)

    # Set up our 'energy'/loss function -- Mean squared distance
    def E_func(params):
        ret = y(xd, params[0], params[1], params[2], params[3],
                params[4])

        n = float(len(ret))
        er = (ret - yd).dot(ret - yd) / n

        if np.isnan(er):
            er = 1e10

        return er

    ####################
    # Heuristic Fitting

    # Initial guess at the parameters:
    p0 = [1.0, 1.0, 1.0, 1.0, 1.0]

    # Step size can greatly affect the solution/convergence.
    mystep = MyStepper(0.1)
    mybound = MyBounds()

    ret = basinhopping(E_func, p0, take_step=mystep, accept_test=mybound)

    ################
    # Show results!

    print "Global Minimum: E(params_fit) = {}".format(ret.fun)
    print "Params: params_fit = [{0}, {1}, {2}, {3}, {4}]".format(*ret.x)

    print "Percent Differences: "
    for pf, (pn, pa) in zip(ret.x, panswer):
        print "{0}: {1:0.2f}%".format(pn, (pa - pf) / pa * 100.0)

    plt.scatter(xd, yd)
    plt.plot(xd, y(xd, *ret.x), c='r')

    fname = "lppl_basinhopping_fit.png"

    save_file = os.path.join('images', fname)

    plt.savefig(save_file)

    plt.show()
