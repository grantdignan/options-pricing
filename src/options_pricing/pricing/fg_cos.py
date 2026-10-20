import numpy as np


def fg_cos(cf, x, taus, r, q, params, strikes, a, b, N):
    taus = taus[:, None]
    r = r[:, None]

    j = np.arange(N)[:, None]
    u = j*np.pi/(b-a)

    strikes = strikes[None, :]
    c = np.maximum(a, np.log(strikes))

    v = np.ones((N, strikes.shape[1]))
    xi = (np.exp(b)*np.cos(np.pi*j[1:]) - np.exp(c)*np.cos(u[1:]*(c-a)) - np.exp(c)*u[1:]*np.sin(u[1:]*(c-a)))/(1+u[1:]**2)
    psi = strikes*np.sin(u[1:]*(c-a))/u[1:]
    v[1:, :] = xi + psi
    v[0, :] = np.exp(b) - np.exp(c) - strikes*(b-c)

    phi = np.ones((len(taus), N), dtype=np.complex128)
    phi[:, 1:] = cf(u[1:, :].T, x, taus, r, q, params)
    phi[:, 0] = 1

    aj = 2*np.real(phi*np.exp(-1j*a*u.T))/(b-a)
    w = np.ones(N)
    w[0] = .5

    return np.exp(-r*taus)*(w*aj @ v)