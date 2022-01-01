using Test
using LazyGrids
using OnlineStatsBase
using MultivariateECDFs


@testset begin
    n = 100
    data_t = tuple.(randn(n), randn(n))
    data_nt = NamedTuple{(:a, :b)}.(data_t)
    
    ecdf_evaluate(data_t, grid(-1:0.2:1, -1:0.2:2))
    ecdf_evaluate(data_t, grid(-1:0.2:1, -1:0.2:2), aggregate=(Orders.Above(), Orders.Below()))
    ecdf_evaluate(data_nt, grid(a=-1:0.2:1, b=-1:0.2:2))
    ecdf_evaluate(data_nt, grid(a=-1:0.2:1, b=-1:0.2:2), aggregate=(a=Orders.Above(),))
    ecdf_evaluate(data_nt, grid(b=-1:0.2:2, a=-1:0.2:1), aggregate=(a=Orders.Above(),))
    ecdf_evaluate(Counter(NamedTuple), data_nt, grid(a=-1:0.2:1, b=-1:0.2:2), aggregate=(a=Orders.Below(), b=Orders.Below()))
    ecdf_evaluate(FTSeries(NamedTuple, Counter(), Variance(), Mean(), transform=x -> x.a), data_nt, grid(a=-1:0.2:1, b=-1:0.2:2), aggregate=(a=Orders.NoAggBelow(),)) .|> value
    @test ecdf_evaluate(data_t, grid(-1:0.2:1, -1:0.2:2), aggregate=(Orders.Below(), Orders.Below())) == ecdf_evaluate(data_t, grid(-1:0.2:1, -1:0.2:2))
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
