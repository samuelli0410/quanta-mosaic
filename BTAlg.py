import pandas as pd
import numpy as np


#We rebalance the portfolio every 5 trading days, holding weights constant in between. This reduces turnover and trading costs while still capturing alpha signals, resulting in a more realistic and robust performance backtest.
# On each rebalance date, we select only the top 50% of stocks ranked by alpha scores, normalize their weights, and hold this basket for the next 5 days.
#We hope this represents a reasonable way to trade on the signal, as it reduces turnover and trading costs while concentrating exposure in the strongest signals.
def compute_alpha_timeseries_hold(
    df,
    stock_col="tickerid",
    date_col="date",
    alpha_col="alpha_fm",
    fwdret_col="d1",
    adv_col="averagedailytradingvolume",
    hold_days=5,
    top_frac=0.5
):
  df = df.copy()
  df = df.drop_duplicates(subset=[date_col, stock_col]).sort_values([date_col, stock_col])
  df = df.replace([np.inf, -np.inf], np.nan)
  # df = df.dropna(subset=[alpha_col, adv_col])

  # Ensure dates are sorted
  dates = pd.Index(pd.to_datetime(df[date_col].unique())).sort_values()

  # map each date -> its anchor date (the first date in its 5-day bucket)
  date_positions = pd.Series(np.arange(len(dates)), index=dates)
  anchor_idx = (date_positions // hold_days) * hold_days
  anchors = dates[anchor_idx.values]
  date_map = pd.DataFrame({date_col: dates, "anchor_date": anchors})

  # Attach the anchor_date map to the full panel
  df[date_col] = pd.to_datetime(df[date_col])
  df = df.merge(date_map, on=date_col, how="left")

  #Build weights ONLY on anchor dates, to prevent excessive rebalancing daily
  rebalance_panel = df[df[date_col] == df["anchor_date"]].copy()

  #Per-day z-score of alpha on anchor days
  g = rebalance_panel.groupby(date_col)
  mu = g[alpha_col].transform("mean")
  sd = g[alpha_col].transform("std").replace(0, np.nan)
  rebalance_panel["alpha_z"] = (rebalance_panel[alpha_col] - mu) / sd
  rebalance_panel = rebalance_panel.dropna(subset=["alpha_z"])

  # Long-only top fraction by alpha_z level on each anchor day
  pct = rebalance_panel.groupby(date_col)["alpha_z"].rank(pct=True, method="first")
  keep = pct >= (1.0 - top_frac)
  rebalance_panel["w_raw"] = np.where(keep, np.maximum(rebalance_panel["alpha_z"], 0.0), 0.0)
  #normalize
  pos_sum = rebalance_panel.groupby(date_col)["w_raw"].transform("sum")
  rebalance_panel["w"] = np.where(pos_sum > 0, rebalance_panel["w_raw"] / pos_sum, 0.0)
  weights = rebalance_panel[[date_col, "anchor_date", stock_col, "w"]].copy()

  #Apply weights across the 5-day holding window
  #For every (date, stock), attach the weight from its period's anchor_date
  #Names not in the anchor basket will have NaN weight -> treated as 0.
  ret_panel = df[[date_col, "anchor_date", stock_col, fwdret_col]].copy()
  ret_panel[fwdret_col] = ret_panel[fwdret_col].astype(float).fillna(0.0)
  ret_panel = ret_panel.merge(weights[[ "anchor_date", stock_col, "w"]], on=["anchor_date", stock_col], how="left")
  #Contribution = w(anchor_date, stock) * d1(date, stock)
  ret_panel["contrib"] = ret_panel["w"].fillna(0.0) * ret_panel[fwdret_col]

  daily_weights = ret_panel[[date_col, stock_col, "w"]].copy().rename(columns={"w": "weight"})
  daily_weights = daily_weights.sort_values([date_col, "weight"], ascending=[True, False]).reset_index(drop=True)

  # Sum across stocks per day
  daily_ret = ret_panel.groupby(date_col)["contrib"].sum().sort_index()
  daily_ret.name = "strategy_ret_gross"
  return daily_ret,daily_weights


