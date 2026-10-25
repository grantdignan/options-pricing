import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from options_pricing.pricing import calibration, fo_cos
from options_pricing.models import black_scholes, heston

mkt = pd.read_csv('test/mkt.csv')
taus = mkt['tau'].to_numpy()
mkt = mkt.drop(columns='tau')
strikes = mkt.columns.to_numpy(dtype=float)

vegas = pd.read_csv('test/vegas.csv').drop(columns='tau')
vegas = np.maximum(vegas.values, 1e-4)


cf = heston.heston_charfun
s0 = 7712.57
x = np.log(s0)
r = .040799
bounds = [(1e-4, 1.0), (1e-4, 1.0), (1e-2, 20.0), (1e-2, 3.0), (-1.0, 0.0)]
x0 = [0.065025, 0.065025, 3.0, 0.8, -0.75]
N = 64

if __name__ == '__main__':
    de = calibration.global_opt(
        mkt.values,
        vegas**2,
        taus,
        strikes,
        cf,
        x,
        r,
        0,
        bounds,
        x0,
        None,
        fo_cos.heston_bounds,
        N
    )
    print('DE runs: ', de.nit)

    result = calibration.local_opt(
        mkt.values,
        vegas**2,
        taus,
        strikes,
        cf,
        x,
        r,
        0,
        bounds,
        x0,
        None,
        fo_cos.heston_bounds,
        N
    )

    print('Loss: ', result.fun)
    print('Local Runs: ', result.nit)
    print(result.x)


    L = 12

    strike_grid = np.arange(s0-150, s0+150, 0.5)

    a, b = fo_cos.heston_bounds(x, taus, r, 0, result.x, L)
    lnk, j = fo_cos.precompute(strike_grid, N)

    calls = fo_cos.fo_cos(cf, x, taus, r, 0, result.x, lnk, strike_grid, j, a, b)
    ivs_model = black_scholes.iv_solve(calls, s0, strike_grid, taus, r)
    ivs_mkt = black_scholes.iv_solve(mkt.values, s0, strikes, taus, r)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(projection='3d')

    X, Y = np.meshgrid(np.log(s0/strike_grid), taus)
    Z = ivs_model

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', linewidth=0, antialiased=True, shade=False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor('0.85')
        axis._axinfo['grid']['color'] = (0.82, 0.82, 0.82, 0.55)
        axis._axinfo['grid']['linewidth'] = 0.45

        ax.set_xlabel('Log-moneyness', labelpad=10)
        ax.set_ylabel('Time to maturity (years)', labelpad=10)
        ax.set_zlabel('Implied volatility', labelpad=8)

        fig.savefig("implied_volatility_surface.png", dpi=600)
        plt.show()