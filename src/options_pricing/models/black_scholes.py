import numpy as np
from scipy.stats import norm
from scipy.optimize import newton


def charfun(u, x, tau, r, q, params):
    i = 1j
    sigma = params[0]

    psi1 = tau*i*u*(r - q - 0.5*sigma**2)
    psi2 = -0.5*(sigma**2)*(u**2)*tau

    return np.exp(i*u*x + psi1 + psi2)


def call(s0, k, tau, r, sigma):
    d1 = np.log(s0/k) + tau*(r + 0.5*sigma**2)/(sigma*np.sqrt(tau))
    d2 = d1 - sigma*np.sqrt(tau)

    return s0*norm.cdf(d1) - k*np.exp(-r*tau)*norm.cdf(d2)

def vega_fun(s0, k, tau, r, sigma):
    d1 = np.log(s0/k) + tau*(r + 0.5*sigma**2)/(sigma*np.sqrt(tau))

    return s0*np.sqrt(tau)*norm.pdf(d1)

def iv_solve(c, s0, k, tau, r, tol=1e-8, max_iter=200):
    tau = tau[:, None]

    active = np.ones_like(c, dtype=bool)

    s0 = np.broadcast_to(s0, c.shape)
    tau = np.broadcast_to(tau, c.shape)
    k = np.broadcast_to(k, c.shape)
    sigma = np.ones_like(c)*0.2

    for i in range(max_iter):
        if not np.any(active):
            break

        c_act = c[active]
        s0_act = s0[active]
        tau_act = tau[active]
        k_act = k[active]
        sigma_act = sigma[active]

        vega = vega_fun(s0_act, k_act, tau_act, r, sigma_act)
        vega = np.clip(vega, 1e-10, np.inf)

        c_bs = call(s0_act, k_act, tau_act, r, sigma_act)

        diff = c_bs - c_act

        sigma[active] = sigma_act - diff/vega

        active[active] = np.abs(diff) >= tol

    return sigma