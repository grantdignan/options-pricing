import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import NonlinearConstraint

from options_pricing.pricing import calibration, fo_cos
from options_pricing.models import heston, black_scholes

mkt = pd.read_csv('test/mkt.csv')
taus = mkt['tau'].to_numpy()
mkt = mkt.drop(columns='tau')
strikes = mkt.columns.to_numpy(dtype=float)

vegas = pd.read_csv('test/vegas.csv').drop(columns='tau')
vegas = np.maximum(vegas.values, 1e-3)


cf = heston.charfun
cos_bounds = fo_cos.heston_bounds
s0 = 7722.6
x = np.log(s0)
r = .0404
bounds = [(1e-4, 1.0), (1e-4, 1.0), (1e-2, 50.0), (1e-2, 10.0), (-1.0, 0.0)]
x0 = [0.06, 0.06, 20.0, 2.5, -0.75]
N = 64

if __name__ == '__main__':
    # feller = NonlinearConstraint(heston.feller, 0.0, np.inf)

    # de = calibration.global_opt(
    #     mkt.values,
    #     vegas**2,
    #     taus,
    #     strikes,
    #     cf,
    #     x,
    #     r,
    #     0,
    #     bounds,
    #     x0,
    #     None,
    #     cos_bounds,
    #     N
    # )
    # print('DE runs: ', de.nit)
    # print(de.x)

    # result = calibration.local_opt(
    #     mkt.values,
    #     vegas**2,
    #     taus,
    #     strikes,
    #     cf,
    #     x,
    #     r,
    #     0,
    #     bounds,
    #     de.x,
    #     None,        
    #     cos_bounds,
    #     N
    # )

    # print('Loss: ', result.fun)
    # print('Local Runs: ', result.nit)
    # print(result.x)
    params = [ 0.02052253,  0.03095877, 19.8645686,   2.07529896, -0.51524545]


    L = 12

    strike_grid = np.arange(s0-150, s0+150, 0.5)

    a, b = cos_bounds(x, taus, r, 0, params, L)
    lnk, j = fo_cos.precompute(strike_grid, N)

    rhos = fo_cos.rho(cf, x, taus, r, 0, params, lnk, strike_grid, j, a, b)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection='3d')

    X, Y = np.meshgrid(np.log(strike_grid/s0), taus)
    Z = rhos
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', linewidth=0, antialiased=True, shade=False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('0.85')
        axis._axinfo['grid']['color'] = (0.82, 0.82, 0.82, 0.55)
        axis._axinfo['grid']['linewidth'] = 0.45

        ax.set_xlabel('Log-moneyness', labelpad=10)
        ax.set_ylabel('Time to maturity (years)', labelpad=10)
        ax.set_zlabel('Implied volatility', labelpad=8)

        plt.show()