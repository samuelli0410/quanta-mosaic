# As a PM, I would mostly care about whether a signal can make money in a portfolio, survive real-world dynamics, and scale, not necessarily if it is just statistically significant.

# Thus, my metrics would be:
# - sharpe ratio: captures return per unit of risk. This allows me to evaluate if the signal produces consistent, risk-adjusted excess returns, not just noisy profits.
# - Turnover-adjusted Sharpe (after costs): captures results after considering trading frictions, such as slippage and commissions. Tests if it can survive real world dynamics.
# - Lagged IC (information coefficient) curve: captures how quickly the signal decays. A signal that only works at t+1 but vanishes (or reverses) at t+5 may not be robust or scalable, so this allows me to investigate signal decay.

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr


def sharpe_ratio(daily_ret):
  daily_ret = pd.Series(daily_ret).dropna()
  mean_ret = daily_ret.mean()
  vol_ret  = daily_ret.std()
  sharpe = (mean_ret / vol_ret)*np.sqrt(252)
  return sharpe

# sharpe = sharpe_ratio(realized_ts)
# print("Sharpe Ratio:", sharpe)











# we estimate 10 bps per day as we assume 5 bps per side for liquid stocks, so 10 bps round trip of buy and sell.
def turnover_adjusted_sharpe_from_weights(daily_weights, daily_ret, date_col="date", stock_col="tickerid",
                        hold_days=5, cost_bps=10):

    daily_weights = daily_weights.copy()
    daily_weights[date_col] = pd.to_datetime(daily_weights[date_col])
    dates = pd.Index(sorted(daily_weights[date_col].unique()))
    pos = pd.Series(range(len(dates)), index=dates)
    anchors = dates[(pos // hold_days) * hold_days]
    date_map = pd.DataFrame({date_col: dates, "anchor_date": anchors})
    w = daily_weights.merge(date_map, on=date_col, how="left")

    # snapshot portfolio only on anchor days
    w_anchor = w[w[date_col] == w["anchor_date"]]
    pivot = w_anchor.pivot_table(index=stock_col, columns=date_col, values="weight", fill_value=0.0)
    cols = pivot.columns.sort_values()

    # compute turnover between consecutive anchor snapshots
    turnover_anchor = {}
    for i in range(1, len(cols)):
        t_prev, t_now = cols[i-1], cols[i]
        turnover_anchor[t_now] = (pivot[t_now] - pivot[t_prev]).abs().sum()

    turnover_anchor = pd.Series(turnover_anchor)

        # expand into daily series (spiky on rebalance days)
    daily_turnover = pd.Series(0.0, index=dates)
    daily_turnover.loc[turnover_anchor.index] = turnover_anchor

    # costs
    cost_perc = cost_bps / 1e4
    daily_cost = daily_turnover * cost_perc
    daily_ret_net = daily_ret - daily_cost

    # sharpe
    vol = daily_ret.std()
    sharpe_gross = daily_ret.mean() / vol * np.sqrt(252) if vol > 0 else np.nan
    sharpe_net   = daily_ret_net.mean() / vol * np.sqrt(252) if vol > 0 else np.nan

    return {
        "Gross Sharpe": sharpe_gross,
        "After-cost Sharpe": sharpe_net,
        "Mean daily turnover": daily_turnover.mean(),
        "Mean daily cost": daily_cost.mean()
    }

# turnover_adjusted_sharpe_from_weights(daily_weights, realized_ts)















def lagged_ic_curve(
    df,
    alpha_col="alpha_fm",
    ret_col="d1",
    date_col="date",
    stock_col="tickerid",
    max_lag=10,
    method="spearman"
):


  d = df[[date_col, stock_col, alpha_col]].dropna(subset=[alpha_col]).copy()
  d = d.merge(Full_df[["tickerid","date","d1"]], on = ["tickerid","date"], how = "left")
  d = d.sort_values([stock_col, date_col])

  # Precompute forward k-day returns via per-ticker shift,
  rets_by_lag = {}
  for k in range(0, max_lag + 1):
    rets_by_lag[k] = d.groupby(stock_col, sort=False)[ret_col].shift(-k)

  ic_daily = {}
  for k in range(0, max_lag + 1):
    dk = d.copy()
    dk["fwd_ret_k"] = rets_by_lag[k]

    dk = dk.dropna(subset=[alpha_col, "fwd_ret_k"])

    # per-day cross-sectional correlation
    def _cs_corr(g):
      x = g[alpha_col].to_numpy()
      y = g["fwd_ret_k"].to_numpy()
      if len(x) < 5:
        return np.nan
      if method == "spearman":
        r, _ = spearmanr(x, y)
      else:
        r, _ = pearsonr(x, y)
      return r

    ic_by_day = dk.groupby(date_col, sort=False).apply(_cs_corr, include_groups=False).astype(float)
    ic_daily[k] = ic_by_day

  rows = []
  for k, ser in ic_daily.items():
    ser = ser.dropna()
    n = ser.size
    mu = ser.mean() if n else np.nan
    sd = ser.std(ddof=1) if n > 1 else np.nan
    tstat = mu / (sd / np.sqrt(n)) if (n > 1 and sd > 0) else np.nan
    rows.append({"lag": k, "mean_ic": mu, "tstat_ic": tstat, "n_days": n})
  ic_summary = pd.DataFrame(rows).set_index("lag")

  return ic_daily, ic_summary




