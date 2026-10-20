import numpy as np


def bates_charfun(u, x, tau, r, q, params):
    i = 1j
    v0, theta, kappa, sigma, rho, lambd, muj, sigmaj = params

    a = rho*sigma*i*u
    d = np.sqrt((a - kappa)**2 + (sigma**2)*(i*u + u**2))
    g = (kappa - a - d)/(kappa - a + d)
    ge = g*np.exp(-d*tau)

    phi1 = i*u*(r - q - lambd*(np.exp(muj + 0.5*sigmaj**2) - 1))*tau
    phi2 = theta*kappa*(sigma**-2)*(-2*np.log((1 - ge)/(1 - g)) + (kappa - a - d)*tau)
    phi3 = v0*(sigma**-2)*(kappa - a - d)*(1 - np.exp(-d*tau))/(1 - ge)
    phi4 = lambd*tau*(np.exp(i*muj*u - 0.5*(sigmaj**2)*(u**2)) - 1)

    return np.exp(i*u*x + phi1 + phi2 + phi3 + phi4)

def feller_fun_bates(params):
    v0, theta, kappa, sigma, rho, lambd, muj, sigmaj = params

    return 2*kappa*theta - sigma**2