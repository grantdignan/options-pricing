# Stochastic Options Pricing

## A vectorized library for pricing options using stochastic models

Models:
* Black Scholes
* Merton Jump Diffusion
* Heston
* Bates

This library uses Fang and Osterlee's COS method for the pricing.

## Installation
Make sure to have uv installed

```
git clone https://github.com/grantdignan/options-pricing

cd options

uv sync
```

## Roadmap
* Optimal COS method bounds
    * Cumulant generating functions for every model
* eSSVI
* Rough Heston
* Quadratic Rough Heston
    * Joint VIX SPX calibration

## References

Black, F., & Scholes, M. (1973). The pricing of options and corporate liabilities. Journal of Political Economy, 81(3), 637–654.

Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. The Review of Financial Studies, 6(2), 327–343.

Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. Journal of Financial Economics, 3(1–2), 125–144.

Bates, D. S. (1996). Jumps and stochastic volatility: Exchange rate processes implicit in Deutsche Mark options. The Review of Financial Studies, 9(1), 69–107.

Fang, F., & Oosterlee, C. W. (2008). A novel pricing method for European options based on Fourier-cosine series expansions. SIAM Journal on Scientific Computing, 31(2), 826–848.