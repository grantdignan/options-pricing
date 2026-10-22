import numpy as np

def calls(puts, strikes, taus, rates, s0):
    taus = taus[:, None]

    return puts + s0 - strikes*np.exp(-rates*taus)

def puts(calls, strikes, taus, rates, s0):
    taus = taus[:, None]

    return calls + s0 - strikes*np.exp(-rates*taus)