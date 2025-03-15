# Descriptive Statistics - Election Market Data

*Generated on 2025-03-15 14:20:40*

## Dataset Overview

- **Dataset dimensions**: 488 rows × 24 columns
- **Numeric features**: 22

## Feature Statistics

### Data Completeness

| feature | count | missing_count | missing_pct |
| --- | --- | --- | --- |
| volumeNum | 488 | 0 | 0.0000 |
| event_commentCount | 488 | 0 | 0.0000 |
| price_volatility | 488 | 0 | 0.0000 |
| price_range | 488 | 0 | 0.0000 |
| final_week_momentum | 488 | 0 | 0.0000 |
| price_fluctuations | 488 | 0 | 0.0000 |
| market_duration_days | 488 | 0 | 0.0000 |
| trading_frequency | 488 | 0 | 0.0000 |
| buy_sell_ratio | 488 | 0 | 0.0000 |
| trading_continuity | 488 | 0 | 0.0000 |
| late_stage_participation | 488 | 0 | 0.0000 |
| volume_acceleration | 488 | 0 | 0.0000 |
| unique_traders_count | 488 | 0 | 0.0000 |
| trader_to_trade_ratio | 488 | 0 | 0.0000 |
| two_way_traders_ratio | 488 | 0 | 0.0000 |
| trader_concentration | 488 | 0 | 0.0000 |
| new_trader_influx | 488 | 0 | 0.0000 |
| comment_per_vol | 488 | 0 | 0.0000 |
| comment_per_trader | 488 | 0 | 0.0000 |
| brier_score | 488 | 0 | 0.0000 |
| log_loss | 488 | 0 | 0.0000 |
| prediction_correct | 488 | 0 | 0.0000 |


### Central Tendency and Dispersion

| feature | mean | median | std | cv |
| --- | --- | --- | --- | --- |
| volumeNum | 11525252.8238 | 709291.7312 | 85755685.6093 | 744.0677 |
| event_commentCount | 7794.1270 | 94.0000 | 38296.8815 | 491.3556 |
| price_volatility | 0.2406 | 0.1709 | 0.2480 | 103.0848 |
| price_range | 0.0711 | 0.0150 | 0.1386 | 195.0422 |
| final_week_momentum | 0.0026 | 0.0000 | 0.0664 | 2551.7727 |
| price_fluctuations | 0.8955 | 0.0000 | 6.3430 | 708.3237 |
| market_duration_days | 169.3770 | 215.0000 | 109.5416 | 64.6732 |
| trading_frequency | 92.8067 | 23.0000 | 262.3527 | 282.6873 |
| buy_sell_ratio | 1.7017 | 1.4302 | 1.1650 | 68.4646 |
| trading_continuity | 0.9231 | 0.6111 | 1.6491 | 178.6557 |
| late_stage_participation | 0.3077 | 0.2262 | 0.3058 | 99.3841 |
| volume_acceleration | 6.1388 | 4.0007 | 5.8370 | 95.0842 |
| unique_traders_count | 2197.3627 | 614.5000 | 5789.0519 | 263.4545 |
| trader_to_trade_ratio | 4.4133 | 3.3896 | 3.2762 | 74.2351 |
| two_way_traders_ratio | 0.3131 | 0.2300 | 0.2302 | 73.5269 |
| trader_concentration | 1.3321 | 1.3067 | 0.1917 | 14.3890 |
| new_trader_influx | 0.2937 | 0.1905 | 0.2984 | 101.6074 |
| comment_per_vol | 0.0006 | 0.0001 | 0.0020 | 357.3580 |
| comment_per_trader | 2.2375 | 0.1195 | 12.4223 | 555.1876 |
| brier_score | 0.0372 | 0.0000 | 0.1193 | 320.4652 |
| log_loss | 0.1216 | 0.0030 | 0.3308 | 272.0119 |
| prediction_correct | 0.9447 | 1.0000 | 0.2289 | 24.2257 |


### Range and Percentiles

| feature | min | p5 | p25 | p75 | p95 | max | range | iqr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| volumeNum | 105948.9794 | 130584.2914 | 242572.6948 | 3433945.7552 | 28449127.6105 | 1531479285.0000 | 1531373336.0206 | 3191373.0605 |
| event_commentCount | 0.0000 | 3.0000 | 14.0000 | 512.0000 | 5184.0000 | 209081.0000 | 209081.0000 | 498.0000 |
| price_volatility | 0.0000 | 0.0000 | 0.0417 | 0.3616 | 0.7677 | 1.3898 | 1.3898 | 0.3198 |
| price_range | 0.0000 | 0.0000 | 0.0040 | 0.0685 | 0.3854 | 0.9100 | 0.9100 | 0.0645 |
| final_week_momentum | -0.4400 | -0.0500 | -0.0019 | 0.0010 | 0.0800 | 0.6300 | 1.0700 | 0.0029 |
| price_fluctuations | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2.0000 | 114.0000 | 114.0000 | 0.0000 |
| market_duration_days | 1.0000 | 16.0000 | 66.0000 | 221.0000 | 300.0000 | 941.0000 | 940.0000 | 155.0000 |
| trading_frequency | 0.7076 | 2.7905 | 9.7710 | 68.8902 | 322.8679 | 3289.0000 | 3288.2924 | 59.1191 |
| buy_sell_ratio | 0.6712 | 0.8554 | 1.1739 | 1.8628 | 2.8051 | 10.9548 | 10.2836 | 0.6888 |
| trading_continuity | 0.1140 | 0.3207 | 0.4751 | 0.9278 | 2.4417 | 31.0000 | 30.8860 | 0.4527 |
| late_stage_participation | 0.0000 | 0.0000 | 0.0267 | 0.4885 | 0.9612 | 1.0000 | 1.0000 | 0.4618 |
| volume_acceleration | 0.0734 | 0.6556 | 2.0751 | 8.0545 | 17.5082 | 29.2218 | 29.1484 | 5.9794 |
| unique_traders_count | 31.0000 | 118.0000 | 295.7500 | 1835.7500 | 8688.6500 | 72183.0000 | 72152.0000 | 1540.0000 |
| trader_to_trade_ratio | 1.2273 | 2.1600 | 2.7890 | 4.6451 | 10.8889 | 24.5931 | 23.3657 | 1.8561 |
| two_way_traders_ratio | 0.0077 | 0.0509 | 0.1350 | 0.4788 | 0.7496 | 0.8703 | 0.8626 | 0.3438 |
| trader_concentration | 0.7377 | 1.0164 | 1.2319 | 1.4451 | 1.6793 | 1.8700 | 1.1323 | 0.2131 |
| new_trader_influx | 0.0000 | 0.0000 | 0.0270 | 0.4982 | 0.8953 | 1.0000 | 1.0000 | 0.4712 |
| comment_per_vol | 0.0000 | 0.0000 | 0.0000 | 0.0002 | 0.0022 | 0.0227 | 0.0227 | 0.0002 |
| comment_per_trader | 0.0000 | 0.0078 | 0.0335 | 0.5537 | 3.0548 | 118.1249 | 118.1249 | 0.5202 |
| brier_score | 0.0000 | 0.0000 | 0.0000 | 0.0036 | 0.2879 | 0.9312 | 0.9312 | 0.0036 |
| log_loss | 0.0010 | 0.0010 | 0.0010 | 0.0619 | 0.7690 | 3.3524 | 3.3514 | 0.0609 |
| prediction_correct | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |


### Distribution Shape

| feature | skewness | kurtosis |
| --- | --- | --- |
| volumeNum | 15.0421 | 242.3138 |
| event_commentCount | 5.0666 | 23.6957 |
| price_volatility | 1.5302 | 2.6307 |
| price_range | 3.1155 | 10.6023 |
| final_week_momentum | 1.1879 | 31.5746 |
| price_fluctuations | 13.0602 | 211.4594 |
| market_duration_days | 1.1078 | 6.7388 |
| trading_frequency | 7.6025 | 71.6680 |
| buy_sell_ratio | 5.0508 | 30.9390 |
| trading_continuity | 13.4078 | 228.9757 |
| late_stage_participation | 0.8513 | -0.4238 |
| volume_acceleration | 1.5399 | 1.9611 |
| unique_traders_count | 7.7662 | 76.2588 |
| trader_to_trade_ratio | 3.2876 | 12.3870 |
| two_way_traders_ratio | 0.8118 | -0.5925 |
| trader_concentration | 0.1303 | 0.6353 |
| new_trader_influx | 0.8447 | -0.4860 |
| comment_per_vol | 7.1387 | 61.1022 |
| comment_per_trader | 7.7961 | 62.1494 |
| brier_score | 4.3294 | 20.5263 |
| log_loss | 4.8102 | 30.4460 |
| prediction_correct | -3.8901 | 13.1326 |


## Interpretation Guide

- **count**: Number of non-missing observations
- **missing_count**: Number of missing values
- **missing_pct**: Percentage of missing values
- **mean**: Average value
- **median**: Middle value (50th percentile)
- **std**: Standard deviation
- **cv**: Coefficient of variation (std/mean × 100)
- **min**: Minimum value
- **max**: Maximum value
- **range**: Difference between max and min
- **p5, p25, p75, p95**: 5th, 25th, 75th, and 95th percentiles
- **iqr**: Interquartile range (p75-p25)
- **skewness**: Measure of asymmetry (0 = symmetric, >0 = right-skewed, <0 = left-skewed)
- **kurtosis**: Measure of "tailedness" (0 = normal, >0 = heavier tails, <0 = lighter tails)

## Distribution Plots

Distribution plots for each feature are available in the `distribution_plots` directory.
