using Test
using LazyGrids
using MultivariateECDFs

@testset begin
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
