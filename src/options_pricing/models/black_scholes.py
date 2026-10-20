import numpy as np
from scipy.stats import norm


def bs_charfun(u, x, tau, r, q, params):
    i = 1j
    sigma = params[0]

    phi1 = tau*i*u*(r - q - 0.5*sigma**2)
    phi2 = -0.5*(sigma**2)*(u**2)*tau

    return np.exp(i*u*x + phi1 + phi2)


def iv(c, s0, tau, r, K, max_iter=200, tol=1e-5):
    sigma0 = np.ones_like(c)

    for i in range(max_iter):
        d1 = (np.log(s0/K) + tau*(r + .5*sigma0))/(sigma0*np.sqrt(tau))
        d2 = d1 - sigma0*np.sqrt(tau)
        c_bs = s0*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)

        vega = s0*np.sqrt(tau)*norm.pdf(d1)
        sigma = sigma0 - (c_bs - c)/vega

        if (abs(sigma - sigma0) < tol).all():
            break

        sigma0 = sigma

    return sigma