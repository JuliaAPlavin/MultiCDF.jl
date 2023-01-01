using Base: NamedTuple
import Random
using Test
using RectiGrids
using OnlineStatsBase
using StableRNGs
using StaticArrays
import StatsBase
using Accessors
using MultiCDFs


@testset "single point" begin
    rng = StableRNG(123)
    n = 100
    data_t2 = tuple.(randn(rng, n), randn(rng, n))
    data_sv2 = SVector.(data_t2)
    data_nt2 = NamedTuple{(:a, :b)}.(data_t2)
    data_r = getindex.(data_t2, 1)

    @testset "empty" begin
        m_ecdf = ecdf(NamedTuple{(:a,)}[])
    end

    @testset "1d" begin
        @testset "numbers" begin
            m_ecdf = ecdf(data_r)
            sb_ecdf = StatsBase.ecdf(data_r)

            @test m_ecdf.(-2:0.5:2) ≈ sb_ecdf(-2:0.5:2)
            @test m_ecdf.(data_r) ≈ sb_ecdf(data_r)
            xs = randn(10)
            @test m_ecdf.(xs) ≈ sb_ecdf(xs)
            @test m_ecdf(-Inf) == 0
            @test m_ecdf(Inf) == 1

            @test_throws MethodError m_ecdf((0.,))
            @test_throws MethodError m_ecdf(SVector(0.,))
            
            @testset "signs" begin
                @test m_ecdf(<=(-Inf)) == 0
                @test m_ecdf(>=(-Inf)) == 1
                @test m_ecdf(0) == m_ecdf(<=(0)) == 1 - m_ecdf(>=(0)) == 0.47
            end
        end

        @testset "tuples" begin
            m_ecdf = ecdf(tuple.(data_r))
            sb_ecdf = StatsBase.ecdf(data_r)
            @test m_ecdf.(tuple.(-2:0.5:2)) ≈ sb_ecdf(-2:0.5:2)
            @test m_ecdf.(SVector.(-2:0.5:2)) ≈ sb_ecdf(-2:0.5:2)
            @test_throws MethodError m_ecdf(0.)
            @test_throws MethodError m_ecdf((a=0.,))
        end

        @testset "namedtuples" begin
            m_ecdf = ecdf(NamedTuple{(:a,)}.(tuple.(data_r)))
            sb_ecdf = StatsBase.ecdf(data_r)
            @test m_ecdf.(NamedTuple{(:a,)}.(tuple.(-2:0.5:2))) ≈ sb_ecdf(-2:0.5:2)
            @test m_ecdf((a=Inf,)) == 1
            @test_throws MethodError m_ecdf(0.)
            @test_throws MethodError m_ecdf((0.,))
            @test_throws MethodError m_ecdf(SVector(0.))
            @test_throws ErrorException m_ecdf((b=0.,))
        end

        @testset "svectors" begin
            m_ecdf = ecdf(SVector.(data_r))
            sb_ecdf = StatsBase.ecdf(data_r)
            @test m_ecdf.(SVector.(-2:0.5:2)) ≈ sb_ecdf(-2:0.5:2)
            @test m_ecdf.(tuple.(-2:0.5:2)) ≈ sb_ecdf(-2:0.5:2)
            @test_throws MethodError m_ecdf(0.)
            @test_throws MethodError m_ecdf((a=0.,))
        end
    end

    @testset "2d" begin
        sb_ecdf1 = StatsBase.ecdf(getindex.(data_t2, 1))
        sb_ecdf2 = StatsBase.ecdf(getindex.(data_t2, 2))

        @testset "tuples" begin
            m_ecdf = ecdf(data_t2)
            @test m_ecdf.(tuple.(-2:0.5:2, Inf)) ≈ sb_ecdf1(-2:0.5:2)
            @test m_ecdf.(tuple.(Inf, -2:0.5:2)) ≈ sb_ecdf2(-2:0.5:2)
            @test m_ecdf((-Inf, -Inf)) == m_ecdf((0., -Inf)) == m_ecdf((Inf, -Inf)) == m_ecdf((-Inf, 0.)) == m_ecdf((-Inf, Inf)) == 0
            @test m_ecdf((Inf, Inf)) == 1
        end

        @testset "namedtuples" begin
            m_ecdf = ecdf(data_nt2)
            @test m_ecdf.(NamedTuple{(:a, :b)}.(tuple.(-2:0.5:2, Inf))) ≈ sb_ecdf1(-2:0.5:2)
            @test m_ecdf.(NamedTuple{(:a, :b)}.(tuple.(Inf, -2:0.5:2))) ≈ sb_ecdf2(-2:0.5:2)
            @test m_ecdf((a=-Inf, b=-Inf)) == m_ecdf((a=0., b=-Inf)) == m_ecdf((a=Inf, b=-Inf)) == m_ecdf((a=-Inf, b=0.)) == m_ecdf((a=-Inf, b=Inf)) == 0
            @test m_ecdf((a=Inf, b=Inf)) == 1
            @test m_ecdf((b=1, a=0)) == m_ecdf((a=0, b=1))
            @test_throws ErrorException m_ecdf((a=0., c=0.))
            
            @testset "signs" begin
                @test m_ecdf((a= <=(-Inf), b= <=(-Inf))) == 0
                @test m_ecdf((a= >=(-Inf), b= >=(-Inf))) == 1
                @test m_ecdf((a= >=(-Inf), b= <=(-Inf))) == 0
                @test m_ecdf((a=0, b=0)) +
                        m_ecdf((a= >=(0), b= >=(0))) +
                        m_ecdf((a= >=(0), b=0)) +
                        m_ecdf((a=0, b= >=(0))) == 1
            end
        end
    end
end

@testset "grid" begin
    rng = StableRNG(123)
    n = 100
    data_t = tuple.(randn(rng, n), randn(rng, n))
    data_nt = NamedTuple{(:a, :b)}.(data_t)
    
    ecdf_t = ecdf(data_t)
    ecdf_nt = ecdf(data_nt)
    
    e1 = ecdf_t.(grid(-1:0.5:1, -1:1:2), count)
    @test e1 == [2 6 9 13; 4 12 22 28; 7 21 38 46; 9 27 50 58; 13 40 70 79]
    @test e1[1, 3] == 9
    @test e1(0, -1) == 7
    
    e2 = (@set ecdf_t.signs[1] = >=).(grid(-1:0.5:1, -1:1:2), count)
    @test e2 == [15 47 78 85; 13 41 65 70; 10 32 49 52; 8 26 37 40; 4 13 17 19]

    e3 = ecdf_nt.(grid(a=-1:0.5:1, b=-1:1:2), count)
    @test e3 == e1
    @test e3[1, 3] == 9
    @test e3(a=0, b=-1) == 7
    @test e3(b=-1, a=0) == 7

    e4 = (@set ecdf_nt.signs.a = >=).(grid(a=-1:0.5:1, b=-1:1:2), count)
    @test e4 == e2

    e5 = (@set ecdf_nt.signs.a = >=).(grid(b=-1:1:2, a=-1:0.5:1), count)
    @test e5 == permutedims(e4)

    @test ecdf_nt.(grid(a=-1:0.5:1, b=-1:1:2)) == ecdf_nt.(grid(a=-1:0.5:1, b=-1:1:2), count) ./ 100

    e6 = ecdf_nt.(grid(a=-1:0.5:1, b=-1:1:2), Counter(NamedTuple))
    @test value.(e6) == e1
end


import Aqua
import CompatHelperLocal as CHL
@testset begin
    CHL.@check()
    Aqua.test_ambiguities(MultiCDFs, recursive=false)
    Aqua.test_unbound_args(MultiCDFs)
    Aqua.test_undefined_exports(MultiCDFs)
    Aqua.test_stale_deps(MultiCDFs)
end
