import numpy as np


def precompute(k, N):
    log_strike = np.log(k)
    j = np.arange(N)[None, :, None]

    return log_strike, j

def call(cf, x, tau, r, q, params, lnk, k, j, a, b):
    N = j.shape[1]
    tau = tau[:, None, None]

    b = b[:, None, None]
    a = a[:, None, None]
    u = j*np.pi/(b-a)

    k = k[None, None, :]
    c = np.maximum(a, lnk)

    v = np.empty((tau.shape[0], N, k.shape[2]))

    uj = u[:, 1:, :]
    i1 = (np.exp(b)*((-1)**j[:, 1:, :]) - uj*np.exp(c)*np.sin(uj*(c-a)) - np.exp(c)*np.cos(uj*(c-a)))/(1+uj**2)
    i2 = (k*np.sin(uj*(c-a)))/uj

    v[:, 1:, :] = np.real(i1 + i2)
    v[:, 0, :] = np.real(np.exp(b) - np.exp(c) - k*(b-c))[:, 0, :]

    phi = np.empty((tau.shape[0], 1, N), dtype=np.complex128)
    phi[:, :, 1:] = cf(uj.transpose(0, 2, 1), x, tau, r, q, params)
    phi[:, :, 0] = 1.0

    aj = 2*np.real(phi*np.exp(-1j*a*u.transpose(0, 2, 1)))/(b-a)
    w = np.ones_like(aj)
    w[:, :, 0] = 0.5

    calls = np.squeeze(np.exp(-r*tau)*(w*aj @ v))

    return np.real(calls)

def delta(cf, x, tau, r, q, params, lnk, k, j, a, b):
    i = 1j
    N = j.shape[1]
    tau = tau[:, None, None]

    b = b[:, None, None]
    a = a[:, None, None]
    u = j*np.pi/(b-a)

    k = k[None, None, :]
    c = np.maximum(a, lnk)

    v = np.empty((tau.shape[0], N, k.shape[2]))

    uj = u[:, 1:, :]
    i1 = (np.exp(b)*((-1)**j[:, 1:, :]) - uj*np.exp(c)*np.sin(uj*(c-a)) - np.exp(c)*np.cos(uj*(c-a)))/(1+uj**2)
    i2 = (k*np.sin(uj*(c-a)))/uj

    v[:, 1:, :] = np.real(i1 + i2)
    v[:, 0, :] = np.real(np.exp(b) - np.exp(c) - k*(b-c))[:, 0, :]

    phi = np.empty((tau.shape[0], 1, N), dtype=np.complex128)
    phi[:, :, 1:] = cf(uj.transpose(0, 2, 1), x, tau, r, q, params)
    phi[:, :, 0] = 1.0

    u = u.transpose(0, 2, 1)
    aj = 2*np.real(phi*np.exp(-i*a*u)*i*u)/(b-a)
    w = np.ones_like(aj)
    w[:, :, 0] = 0.5

    d = np.squeeze(np.exp(-r*tau)*(w*aj @ v)/np.exp(x))

    return np.real(d)

def gamma(cf, x, tau, r, q, params, lnk, k, j, a, b):
    i = 1j
    N = j.shape[1]
    tau = tau[:, None, None]

    b = b[:, None, None]
    a = a[:, None, None]
    u = j*np.pi/(b-a)

    k = k[None, None, :]
    c = np.maximum(a, lnk)

    v = np.empty((tau.shape[0], N, k.shape[2]))

    uj = u[:, 1:, :]
    i1 = (np.exp(b)*((-1)**j[:, 1:, :]) - uj*np.exp(c)*np.sin(uj*(c-a)) - np.exp(c)*np.cos(uj*(c-a)))/(1+uj**2)
    i2 = (k*np.sin(uj*(c-a)))/uj

    v[:, 1:, :] = np.real(i1 + i2)
    v[:, 0, :] = np.real(np.exp(b) - np.exp(c) - k*(b-c))[:, 0, :]

    phi = np.empty((tau.shape[0], 1, N), dtype=np.complex128)
    phi[:, :, 1:] = cf(uj.transpose(0, 2, 1), x, tau, r, q, params)
    phi[:, :, 0] = 1.0

    u = u.transpose(0, 2, 1)
    aj = 2*np.real(phi*np.exp(-i*a*u)*((-(u)**2)-i*u))/(b-a)
    w = np.ones_like(aj)
    w[:, :, 0] = 0.5

    g = np.squeeze(np.exp(-r*tau)*(w*aj @ v)/(np.exp(x)**2))

    return np.real(g)

def bs_bounds(x, tau, r, q, params, L=12):
    i = 1j
    sigma = params[0]

    c1 = x + tau*(r-q-0.5*(sigma**2))
    c2 = (sigma**2)*tau

    a = c1 - L*np.sqrt(np.abs(c2))
    b = c1 + L*np.sqrt(np.abs(c2))

    return a, b

def mjd_bounds(x, tau, r, q, params, L=12):
    i = 1j
    sigma, lambd, muj, sigmaj = params

    du = i*x + tau*i*(r-q-0.5*(sigma**2)-lambd*(np.exp(muj + 0.5*sigmaj**2) - 1))
    duu = tau*((sigma**2)-lambd*((sigmaj**2) + (muj**2)))

    c1 = du/i
    c2 = -duu

    a = c1 - L*np.sqrt(np.abs(c2))
    b = c1 + L*np.sqrt(np.abs(c2))

    return a, b

def heston_bounds(x, tau, r, q, params, L=12):
    v0, theta, kappa, sigma, rho = params

    a = rho*sigma/kappa - 1 - (sigma**2)/(4*(kappa**2))
    b = (sigma**2)/(2*(kappa**2)) - rho*sigma/kappa
    c = -(sigma**2)/(4*(kappa**2))

    etk = np.exp(tau*kappa)
    etkn = 1/etk

    du = (-1/(2*kappa))*(1 - etkn)
    duu = etkn*((a/kappa)*(etk-1) + b*tau + (c/kappa)*(1-etkn))

    cu = (r-q)*tau - 0.5*(theta)*(tau+(etkn-1)/kappa)
    cuu = theta*(a*(tau-(1-etkn)/kappa) + (b/kappa)*(1-etkn*(kappa*tau + 1)) + (c/kappa)*(1-etkn-0.5*(1-np.exp(-2*tau*kappa))))

    c1 = x + cu + v0*du
    c2 = -(cuu + v0*duu)

    a = c1 - L*np.sqrt(np.abs(c2))
    b = c1 + L*np.sqrt(np.abs(c2))

    return a, b

def bates_bounds(x, tau, r, q, params, L=12):
    v0, theta, kappa, sigma, rho, lambd, muj, sigmaj = params

    a = rho*sigma/kappa - 1 - (sigma**2)/(4*(kappa**2))
    b = (sigma**2)/(2*(kappa**2)) - rho*sigma/kappa
    c = -(sigma**2)/(4*(kappa**2))

    etk = np.exp(tau*kappa)
    etkn = 1/etk

    du = (-1/(2*kappa))*(1 - etkn)
    duu = etkn*((a/kappa)*(etk-1) + b*tau + (c/kappa)*(1-etkn))

    cu = (r-q-lambd*(np.exp(muj + 0.5*sigmaj**2) - 1))*tau - 0.5*(theta)*(tau+(etkn-1)/kappa)
    cuu = theta*(a*(tau-(1-etkn)/kappa) + (b/kappa)*(1-etkn*(kappa*tau + 1)) + (c/kappa)*(1-etkn-0.5*(1-np.exp(-2*tau*kappa))))

    mjdu = lambd*tau*muj
    mjduu = -lambd*tau*((sigmaj**2) + (muj**2))

    c1 = x + cu + v0*du + mjdu
    c2 = -(cuu + v0*duu + mjduu)

    a = c1 - L*np.sqrt(np.abs(c2))
    b = c1 + L*np.sqrt(np.abs(c2))

    return a, b