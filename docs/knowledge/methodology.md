# How the numbers are produced

## Forecast
Daily sales per store are forecast by a gradient-boosted tree model trained on calendar,
promotion schedule, holidays, store attributes and competition features. Forecasts used for
anomaly detection are always out-of-sample: each quarter is forecast by a model trained only on
data before that quarter (rolling-origin backtest). Store-week forecast error (WAPE) is about 7%.

## Prediction interval and calibration
Quantile models give a 90% prediction interval per store-day. Raw intervals were too narrow, so
they are widened using the previous quarter's observed errors (conformal calibration). Coverage
after calibration is close to 90% except around Christmas.

## What counts as an anomaly
A store-week is anomalous when actual sales differ from forecast by at least 10% AND most of the
store's open days that week fall outside the calibrated interval in the same direction.
Consecutive anomalous weeks in the same direction form an episode. Episodes are ranked by euro gap.

## Root-cause lenses
1. Traffic vs basket: the sales gap is split exactly into a customer-visits part and a
   spend-per-visit part (log decomposition). "Traffic-driven" means visits explain at least 60%.
2. Dimensional drill-down: which store segments (assortment, competition distance band, Promo2
   enrollment) concentrate the gap, Adtributor-style.
3. Event evidence: store reopened after closure, competitor opened nearby, Promo2 started,
   Easter/Christmas overlap, segment-wide movement (15%+ of same-type stores moved the same way),
   and persistence (gap continues 4 weeks after the episode -> possible level shift).
An episode with no matching evidence is labelled "unexplained": it needs local investigation.

## Known limitations
- The forecaster has no "days until holiday" feature, so it under-forecasts pre-holiday shopping.
- Tree models do not extrapolate trend; in growing periods the forecast runs low, so
  above-forecast episodes are more common than below-forecast ones.
- No weather, local events or margin data.
