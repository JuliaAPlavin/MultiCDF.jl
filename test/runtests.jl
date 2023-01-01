using Base: NamedTuple
import Random
using Test
using RectiGrids
using OnlineStatsBase
using StableRNGs
using StaticArrays
import StatsBase
using Accessors
using MultiCDF


@testset "empty" begin
    m_ecdf = ecdf(NamedTuple{(:a,)}[])
    @test m_ecdf((a=1,)) |> isnan
    @test m_ecdf((a=1,), count) == 0
    @test m_ecdf((a=1,), Counter(NamedTuple)) |> value == 0
end

@testset "numbers" begin
    rng = StableRNG(123)
    data = randn(rng, 100)
    m_ecdf = ecdf(data)
    sb_ecdf = StatsBase.ecdf(data)

    @test m_ecdf.(-2:0.5:2) ≈ sb_ecdf(-2:0.5:2)
    @test m_ecdf.(data) ≈ sb_ecdf(data)
    xs = randn(rng, 10)
    @test m_ecdf.(xs) ≈ sb_ecdf(xs) ≈ [0.24, 0.28, 0.01, 0.63, 0.02, 0.01, 0.54, 0.73, 0.02, 0.7]
    @test m_ecdf(-Inf) == 0
    @test m_ecdf(Inf) == 1

    @test_throws MethodError m_ecdf((0.,))
    @test_throws MethodError m_ecdf(SVector(0.,))
    
    @test m_ecdf(<=(-Inf)) == 0
    @test m_ecdf(>=(-Inf)) == 1
    @test m_ecdf(0) == m_ecdf(<=(0)) == 1 - m_ecdf(>=(0)) == 0.47
    @test m_ecdf(0, count) == m_ecdf(<=(0), count) == 100 - m_ecdf(>=(0), count) == 47
end

@testset for T in [Tuple, SVector]
    rng = StableRNG(123)
    data = T.(Tuple.(eachrow(randn(rng, 100, 2))))

    sb_ecdf1 = StatsBase.ecdf(getindex.(data, 1))
    m_ecdf = ecdf(data)

    @test m_ecdf.(tuple.(-2:0.5:2, Inf)) ≈ sb_ecdf1(-2:0.5:2) ≈ [0.01, 0.02, 0.13, 0.29, 0.47, 0.6, 0.81, 0.89, 0.95]
    @test m_ecdf((-Inf, -Inf)) == m_ecdf((0., -Inf)) == m_ecdf((Inf, -Inf)) == m_ecdf((-Inf, 0.)) == m_ecdf((-Inf, Inf)) == 0
    @test m_ecdf((Inf, Inf)) == 1
    @test m_ecdf((0, 1)) == m_ecdf(T((0, 1))) == m_ecdf((<=(0), <=(1))) == 0.38
    @test m_ecdf(T((0, >=(1))), count) == m_ecdf((<=(0), >=(1)), count) == 9

    @test_throws Exception m_ecdf(())
    @test_throws Exception m_ecdf((a=1, b=2))

    e1 = m_ecdf.(grid(T, -1:0.5:1, -1:1:2), count)
    @test e1 == [2 6 9 13; 4 12 22 28; 7 21 38 46; 9 27 50 58; 13 40 70 79]
    @test e1 == m_ecdf.(grid(T, -1:0.5:1, -1:1:2) |> collect, count)
    @test e1[1, 3] == 9
    @test e1(0, -1) == 7

    e2 = (@set m_ecdf.signs[1] = >=).(grid(T, -1:0.5:1, -1:1:2))
    @test e2 == [15 47 78 85; 13 41 65 70; 10 32 49 52; 8 26 37 40; 4 13 17 19] ./ 100
    @test e2 == (@set m_ecdf.signs[1] = >=).(grid(T, -1:0.5:1, -1:1:2) |> collect)
end

@testset "namedtuples" begin
    rng = StableRNG(123)
    data = NamedTuple{(:a, :b)}.(eachrow(randn(rng, 100, 2)))
    sb_ecdf1 = StatsBase.ecdf(getindex.(data, 1))

    m_ecdf = ecdf(data)
    @test m_ecdf((a=-Inf, b=-Inf)) == m_ecdf((a=0., b=-Inf)) == m_ecdf((a=Inf, b=-Inf)) == m_ecdf((a=-Inf, b=0.)) == m_ecdf((a=-Inf, b=Inf)) == 0
    @test m_ecdf((a=Inf, b=Inf)) == 1
    @test m_ecdf((b=1, a=0)) == m_ecdf((a=0, b=1))
    @test m_ecdf((b=1,)) == 0.87
    @test m_ecdf((b= >=(1),), count) == 13
    @test m_ecdf.(NamedTuple{(:a, :b)}.(tuple.(-2:0.5:2, Inf))) ≈ sb_ecdf1(-2:0.5:2)
    @test m_ecdf.(NamedTuple{(:a,)}.(-2:0.5:2)) ≈ sb_ecdf1(-2:0.5:2)
    @test_throws ErrorException m_ecdf((a=0., c=0.))
    
    @test m_ecdf((a= <=(-Inf), b= <=(-Inf))) == 0
    @test m_ecdf((a= >=(-Inf), b= >=(-Inf)), count) == 100
    @test m_ecdf((a=0, b=0)) +
            m_ecdf((a= >=(0), b= >=(0))) +
            m_ecdf((a= >=(0), b=0)) +
            m_ecdf((a=0, b= >=(0))) == 1

    e3 = m_ecdf.(grid(a=-1:0.5:1, b=-1:1:2), count)
    @test e3[1, 3] == 9
    @test e3(a=0, b=-1) == 7
    @test e3(b=-1, a=0) == 7

    e4 = (@set m_ecdf.signs.a = >=).(grid(a=-1:0.5:1, b=-1:1:2))
    @test e4[1, 1] == m_ecdf((a= >=(-1), b=-1))
    @test e4(-0.5, 0) == m_ecdf((a= >=(-0.5), b=0))
    @test e4(a=-0.5, b=0) == m_ecdf((a= >=(-0.5), b=0))

    e5 = (@set m_ecdf.signs.a = >=).(grid(b=-1:1:2, a=-1:0.5:1), count)
    @test e5 == permutedims(e4) .* 100

    @test m_ecdf.(grid(a=-1:0.5:1, b=-1:1:2)) == m_ecdf.(grid(a=-1:0.5:1, b=-1:1:2) |> collect, count) ./ 100

    e6 = m_ecdf.(grid(a=-1:0.5:1, b=-1:1:2), Counter(NamedTuple))
    @test value.(e6) == e3

    e7 = m_ecdf.(grid(a=-1:0.5:1))
    @test e7(a=0.5) == m_ecdf((a=0.5, b=Inf)) == 0.6
end


import Aqua
import CompatHelperLocal as CHL
@testset begin
    CHL.@check()
    Aqua.test_ambiguities(MultiCDF, recursive=false)
    Aqua.test_unbound_args(MultiCDF)
    Aqua.test_undefined_exports(MultiCDF)
    Aqua.test_stale_deps(MultiCDF)
end
