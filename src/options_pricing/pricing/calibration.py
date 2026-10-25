import numpy as np
from scipy.optimize import differential_evolution, NonlinearConstraint, minimize

from options_pricing.pricing.fo_cos import fo_cos, precompute
from options_pricing.models import heston, bates


def loss_fun(params, mkt_c, w, tau, lnk, mkt_k, cf, x, r, q, j, cos_bounds_fun):
    a, b = cos_bounds_fun(x, tau, r, q, params, L=12)

    model_c = fo_cos(cf, x, tau, r, q, params, lnk, mkt_k, j, a, b)

    valid = np.isfinite(mkt_c) & np.isfinite(model_c) & (w != 0.0)

    errors = w[valid]*((mkt_c[valid] - model_c[valid])**2)

    return np.mean(errors)


def global_opt(mkt_c, w, tau, mkt_k, cf, x, r, q, bounds, x0, constraint, cos_bounds_fun, N):
    mkt_c[mkt_c == 0.0] = np.nan
    lnk, j = precompute(mkt_k, N)

    if constraint is not None:
        result = differential_evolution(
            loss_fun,
            bounds=bounds,
            args=(mkt_c, w, tau, lnk, mkt_k, cf, x, r, q, j, cos_bounds_fun),
            maxiter=40,
            x0=x0,
            constraints=constraint,
            polish=False,
            popsize=5,
            workers=-1
        )
    else:
        result = differential_evolution(
            loss_fun,
            bounds=bounds,
            args=(mkt_c, w, tau, lnk, mkt_k, cf, x, r, q, j, cos_bounds_fun),
            maxiter=40,
            x0=x0,
            polish=False,
            popsize=5,
            workers=-1
        )

    return result

def local_opt(mkt_c, w, tau, mkt_k, cf, x, r, q, bounds, x0, constraint, cos_bounds_fun, N):
    mkt_c[mkt_c == 0.0] = np.nan
    lnk, j = precompute(mkt_k, N)

    if constraint is not None:
        result = minimize(
            loss_fun,
            bounds=bounds,
            args=(mkt_c, w, tau, lnk, mkt_k, cf, x, r, q, j, cos_bounds_fun),
            x0=x0,
            constraints=constraint,
        )
    else:
        result = minimize(
            loss_fun,
            bounds=bounds,
            args=(mkt_c, w, tau, lnk, mkt_k, cf, x, r, q, j, cos_bounds_fun),
            x0=x0,
        )

    return result