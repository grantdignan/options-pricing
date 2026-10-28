import numpy as np


def charfun(u, x, tau, r, q, params):
    i = 1j
    v0, theta, kappa, sigma, rho, lambd, muj, sigmaj = params

    a = rho*sigma*i*u
    d = np.sqrt((a - kappa)**2 + (sigma**2)*(i*u + u**2))
    g = (kappa - a - d)/(kappa - a + d)
    ge = g*np.exp(-d*tau)

    drift = i*u*(r - q - lambd*(np.exp(muj + 0.5*sigmaj**2) - 1))*tau
    C = theta*kappa*(sigma**-2)*(-2*np.log((1 - ge)/(1 - g)) + (kappa - a - d)*tau)
    D = (sigma**-2)*(kappa - a - d)*(1 - np.exp(-d*tau))/(1 - ge)
    mjd = lambd*tau*(np.exp(i*muj*u - 0.5*(sigmaj**2)*(u**2)) - 1)

    return np.exp(i*u*x + drift + C + D*v0 + mjd)

def vega_charfun(u, x, tau, r, q, params):
    i = 1j
    v0, theta, kappa, sigma, rho, lambd, muj, sigmaj = params

    a = rho*sigma*i*u
    d = np.sqrt((a - kappa)**2 + (sigma**2)*(i*u + u**2))
    g = (kappa - a - d)/(kappa - a + d)
    ge = g*np.exp(-d*tau)

    drift = i*u*(r - q - lambd*(np.exp(muj + 0.5*sigmaj**2) - 1))*tau
    C = theta*kappa*(sigma**-2)*(-2*np.log((1 - ge)/(1 - g)) + (kappa - a - d)*tau)
    D = (sigma**-2)*(kappa - a - d)*(1 - np.exp(-d*tau))/(1 - ge)
    mjd = lambd*tau*(np.exp(i*muj*u - 0.5*(sigmaj**2)*(u**2)) - 1)

    return 2*D*np.sqrt(v0)*np.exp(i*u*x + drift + C + D*v0 + mjd)

def theta_charfun(u, x, tau, r, q, params):
    i = 1j
    v0, theta, kappa, sigma, rho, lambd, muj, sigmaj = params

    a = rho*sigma*i*u
    d = np.sqrt((a - kappa)**2 + (sigma**2)*(i*u + u**2))
    g = (kappa - a - d)/(kappa - a + d)
    ge = g*np.exp(-d*tau)

    drift = i*u*(r - q - lambd*(np.exp(muj + 0.5*sigmaj**2) - 1))*tau
    C = theta*kappa*(sigma**-2)*(-2*np.log((1 - ge)/(1 - g)) + (kappa - a - d)*tau)
    D = (sigma**-2)*(kappa - a - d)*(1 - np.exp(-d*tau))/(1 - ge)
    mjd = lambd*tau*(np.exp(i*muj*u - 0.5*(sigmaj**2)*(u**2)) - 1)

    dC = kappa*theta*D + i*u*r
    dD = 0.5*(i*u - u**2) + (kappa - a)*D + 0.5*(sigma**2)*(D**2)
    dmjd = lambd*(np.exp(i*muj*u - 0.5*(sigmaj**2)*(u**2)) - 1)

    return i*u*(r-q) + dC + v0*dD + dmjd, np.exp(i*u*x + drift + C + D*v0 + mjd)

def feller(params):
    v0, theta, kappa, sigma, rho, lambd, muj, sigmaj = params

    return 2*kappa*theta - sigma**2