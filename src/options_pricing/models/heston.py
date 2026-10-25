import numpy as np


def heston_charfun(u, x, tau, r, q, params):
    i = 1j
    v0, theta, kappa, sigma, rho = params

    a = rho*sigma*i*u
    d = np.sqrt((a - kappa)**2 + (sigma**2)*(i*u + u**2))
    g = (kappa - a - d)/(kappa - a + d)
    ge = g*np.exp(-d*tau)

    C = i*u*(r-q)*tau + theta*kappa*(sigma**-2)*(-2*np.log((1 - ge)/(1 - g)) + (kappa - a - d)*tau)
    D = (sigma**-2)*(kappa - a - d)*(1 - np.exp(-d*tau))/(1 - ge)

    return np.exp(i*u*x + C + D*v0)

def feller(params):
    v0, theta, kappa, sigma, rho = params

    return 2*kappa*theta - sigma**2