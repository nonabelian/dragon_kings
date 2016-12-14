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
import pymc as pm
import numpy as np
import scipy.stats as scs
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

    #############
    # PyMC Model
#    o_sd = 40.0
#    o_tau = 1 / o_sd**2
#    m_sd = 5.0
#    m_tau = 1 / m_sd**2
#    A_sd = 200.0
#    A_tau = 1 / A_sd**2
#    C_sd = 200.0
#    C_tau = 1 / C_sd**2
#    t_sd = 40.0
#    t_tau = 1 / t_sd**2

    o_sd = 10.0
    o_tau = 1 / o_sd**2
    m_sd = 10.0
    m_tau = 1 / m_sd**2
    A_sd = 10.0
    A_tau = 1 / A_sd**2
    C_sd = 10.0
    C_tau = 1 / C_sd**2
    t_sd = 10.0
    t_tau = 1 / t_sd**2

    o = pm.Uniform('o', lower=0, upper=200)
    m = pm.Uniform('m', lower=-10, upper=10)
    A = pm.Uniform('A', lower=-1e4, upper=1e4)
    C = pm.Uniform('C', lower=-1e4, upper=1e4)
    t = pm.Uniform('t', lower=1e-5, upper=200)

#    o = pm.HalfNormal('o', tau=o_tau)
#    m = pm.Normal('m', mu=0.0, tau=m_tau)
#    A = pm.Normal('A', mu=0.0, tau=A_tau)
#    C = pm.Normal('C', mu=0.0, tau=C_tau)
#    t = pm.HalfNormal('t', tau=t_tau)

    @pm.deterministic
    def y_mu(o=o, m=m, A=A, C=C, t=t):
        ret = A + t**m * xd**m + C * t**m * xd**m * np.cos(o * np.log(xd))

    y_sd = pm.HalfNormal('y_sd', tau=1)
    y_tau = 1/y_sd**2

    y_obs = pm.Normal('y_obs', mu=y_mu, tau=y_tau, value=yd, observed=True)

    model = pm.Model([y_obs, y_mu, y_sd, t, C, A, m, o])

    mcmc = pm.MCMC(model)
    mcmc.sample(iter=200000, burn=10000, thin=10)

    osamp = mcmc.trace('o')[:]
    msamp = mcmc.trace('m')[:]

    plt.figure(figsize=(10,10))
    plt.plot(osamp, alpha=0.75, c='b')
    plt.plot(msamp, alpha=0.75, c='r')
    
    plt.figure(figsize=(10,10))
    plt.scatter(xd, yd)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(211)
    plt.hist(osamp, bins=25, alpha=0.5)
    ax = fig.add_subplot(212)
    plt.hist(msamp, bins=25, alpha=0.5)
    plt.show()


