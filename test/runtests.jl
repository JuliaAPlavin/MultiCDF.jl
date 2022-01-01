using Test
using LazyGrids
using MultivariateECDFs

@testset begin
    n = 100
    data_t = tuple.(randn(n), randn(n))
    data_nt = NamedTuple{(:a, :b)}.(data_t)
    
    ecdf_evaluate(data_t, grid(-1:0.2:1, -1:0.2:2))
end


import Aqua
import CompatHelperLocal as CHL
@testset begin
    CHL.@check()
    Aqua.test_ambiguities(MultivariateECDFs, recursive=false)
    Aqua.test_unbound_args(MultivariateECDFs)
    Aqua.test_undefined_exports(MultivariateECDFs)
    Aqua.test_stale_deps(MultivariateECDFs)
end
