import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from options_pricing.pricing import parity, calibration, fg_cos
from options_pricing.models import black_scholes, heston

mkt = pd.read_csv('test/mkt.csv')
taus = mkt['tau'].to_numpy()
mkt = mkt.drop(columns='tau')

vegas = pd.read_csv('test/vegas.csv').drop(columns='tau')
vegas = np.maximum(vegas.values, 1e-3)

strikes = mkt.columns.to_numpy(dtype=float)
cf = heston.heston_charfun
x = np.log(7712.57)
r = .040799
bounds = [(1e-4, 1.0), (1e-4, 1.0), (1e-2, 20.0), (1e-2, 3.0), (-1.0, 0.0)]
x0 = [0.065025, 0.065025, 3.0, 0.8, -0.75]

# de = calibration.global_opt(mkt.values, vegas**2, taus, strikes, cf, x, r, 0, bounds, x0)
# result = calibration.min_opt(mkt.values, vegas**2, taus, strikes, cf, x, r, 0, bounds, de.x)

# print(result.fun)
# print(result.x)

params = [ 0.01831349,  0.824739,    0.15196894,  1.08560558, -0.19990609]

strike_grid = np.arange(strikes.min(), strikes.max(), 1)

calls = fg_cos.fg_cos(cf, x, taus, r, 0, params, strike_grid, x-.2, x+.2, 128)
iv_model = black_scholes.solve(calls, 7712.57, taus, r, strike_grid)
iv_mkt = black_scholes.solve(mkt.values, 7712.57, taus, r, strikes)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

X, Y = np.meshgrid(strike_grid, taus[2:])
Z = iv_model[2:, :]

surf = ax.plot_surface(X, Y, Z, cmap='GnBu')

strikes, taus = np.meshgrid(strikes, taus)

# ax.scatter3D(strikes, taus, iv_mkt, c='red', s=10, marker='+')

plt.show()