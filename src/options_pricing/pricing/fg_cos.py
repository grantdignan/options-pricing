import numpy as np


def fg_cos(cf, x, tau, r, q, params, k, a, b, N):
    tau = tau[:, None]

    j = np.arange(N)[:, None]
    u = j*np.pi/(b-a)

    k = k[None, :]
    c = np.maximum(a, np.log(k))

    v = np.ones((N, k.shape[1]))

    uj = u[1:]
    i1 = (np.exp(b)*np.cos(np.pi*j[1:]) - uj*np.exp(c)*np.sin(uj*(c-a)) - np.exp(c)*np.cos(uj*(c-a)))/(1+uj**2)
    i2 = (k*np.sin(uj*(c-a)))/uj

    v[1:, :] = i1 + i2
    v[0, :] = np.exp(b) - np.exp(c) - k*(b-c)

    phi = np.ones((len(tau), N), dtype=np.complex128)
    phi[:, 1:] = cf(u[1:, :].T, x, tau, r, q, params)
    phi[:, 0] = 1

    aj = 2*np.real(phi*np.exp(-1j*a*u.T))/(b-a)
    w = np.ones(N)
    w[0] = .5

    return np.exp(-r*tau)*(w*aj @ v)