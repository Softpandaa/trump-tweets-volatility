"""Event regression of the daily VIX change on its own lag and the tweet regressors."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config


def vix_change(prices):
    """r_t = 100 ln(V_t / V_{t-1}), V_t the VIX close of session t."""
    return (100 * np.log(prices["VIX_close"]).diff()).dropna().rename("r")


def event_regression(r, X):
    """Equation (3) or (5) for each k in LAGS, Newey-West OLS: coefficients and p-values."""
    index = X.index.union(r.index)
    r, X = r.reindex(index), X.reindex(index).fillna(0.0)
    coef, pval = {}, {}
    for k in config.LAGS:
        y, lag = r.shift(-k).rename("y"), r.shift(1 - k).rename("lag")
        df = pd.concat([y, lag, X], axis=1).dropna()
        fit = sm.OLS(df["y"], sm.add_constant(df.drop(columns="y"))).fit(
            cov_type="HAC", cov_kwds={"maxlags": config.HAC_LAGS})
        coef[k], pval[k] = fit.params, fit.pvalues
    return pd.DataFrame(coef), pd.DataFrame(pval)
