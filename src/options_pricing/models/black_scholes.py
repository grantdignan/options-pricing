import numpy as np
from scipy.stats import norm
from scipy.optimize import newton


def bs_charfun(u, x, tau, r, q, params):
    i = 1j
    sigma = params[0]

    phi1 = tau*i*u*(r - q - 0.5*sigma**2)
    phi2 = -0.5*(sigma**2)*(u**2)*tau

    return np.exp(i*u*x + phi1 + phi2)


def bs_call(s0, tau, r, K, sigma):
    sigma = np.clip(sigma, 1e-5, 5.0)

    d1 = (np.log(s0/K) + tau*(r + .5*(sigma**2)))/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)
    c_bs = s0*norm.cdf(d1) - K*np.exp(-r*tau)*norm.cdf(d2)

    return c_bs

def vega_fun(sigma, c, s0, tau, r, K):
    sigma = np.clip(sigma, 1e-5, 5.0)

    d1 = (np.log(s0/K) + tau*(r + .5*(sigma**2)))/(sigma*np.sqrt(tau))

    vega = s0*np.sqrt(tau)*norm.pdf(d1)
    vega = np.clip(vega, 1e-6, np.inf)

    return vega

def obj(sigma, c, s0, tau, r, K):
    return bs_call(s0, tau, r, K, sigma) - c

def solve(c, s0, tau, r, K):
    tau = tau[:, None]
    x0 = np.ones_like(c)*0.2

    iv = newton(obj, x0, vega_fun, (c, s0, tau, r, K), 1e-5, 1000)

    return iv