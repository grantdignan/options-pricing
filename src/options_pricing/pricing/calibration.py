import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint, minimize

from options_pricing import fg_cos, heston, bates


def loss_fun(params, mkt_c, taus, mkt_k, cf, x, r, q):
    model_c = fg_cos(cf, x, taus, r, q, params, mkt_k, x-.5, x+.5, 512)

    mkt_c[mkt_c == 0.0] = np.nan
    valid = np.isfinite(mkt_c) & np.isfinite(model_c)

    errors = mkt_c[valid] - model_c[valid]

    return np.mean(errors**2)


def global_opt(mkt_c, taus, mkt_k, cf, x, r, q, bounds, x0):
    if cf == heston.heston_charfun:
        feller_con = NonlinearConstraint(heston.feller_fun_heston, 0.0, np.inf)

        result = differential_evolution(loss_fun, bounds=bounds, args=(mkt_c, taus, mkt_k, cf, x, r, q), polish=False, constraints=feller_con, x0=x0)

    if cf == bates.bates_charfun:
        feller_con = NonlinearConstraint(bates.feller_fun_bates, 0.0, np.inf)

        result = differential_evolution(loss_fun, bounds=bounds, args=(mkt_c, taus, mkt_k, cf, x, r, q), polish=False, x0=x0)

    else:
        result = differential_evolution(loss_fun, bounds=bounds, args=(mkt_c, taus, mkt_k, cf, x, r, q), polish=False, x0=x0)

    return result

def min_opt(mkt_c, taus, mkt_k, cf, x, r, q, bounds, x0):
    result = minimize(loss_fun, bounds=bounds, args=(mkt_c, taus, mkt_k, cf, x, r, q), x0=x0)

    return result