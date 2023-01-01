# MultiCDF.jl

Empirical Multivariate Cumulative Distribution Function.

Main features of `MultiCDF.jl`:
- Provides a way to represent multivariate ECDFs, calculate their values at arbitrary points;
- Includes an efficient algorithm to compute ECDF at a grid of points at once;
- Supports a variety of point types: `Tuple`s, `NamedTuple`s, `StaticVector`s _(doesn't depend on `StaticArrays` itself)_;
- Can calculate marginal ECDFs taking a subset of parameters into account;
- Computes generalized ECDFs: arbitrary signs (`<=` / `>=` / `<` / `>`), arbitrary `OnlineStats` statistics instead of simple counting.

See a [Pluto notebook](https://aplavin.github.io/MultiCDF.jl/test/examples.html) for examples and benchmarks.
