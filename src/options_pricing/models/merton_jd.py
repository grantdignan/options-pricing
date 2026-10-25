import numpy as np


def mjd_charfun(u, x, tau, r, q, params):
    i = 1j
    sigma, lambd, muj, sigmaj = params

    drift = tau*i*u*(r - q - 0.5*sigma**2 - lambd*(np.exp(muj + 0.5*sigmaj**2) - 1))
    psi1 = -0.5*(sigma**2)*(u**2)*tau
    psi2 = lambd*tau*(np.exp(i*muj*u - 0.5*(sigmaj**2)*(u**2)) - 1)

    return np.exp(i*u*x + drift + psi1 + psi2)