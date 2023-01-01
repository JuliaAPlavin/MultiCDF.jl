using Base: NamedTuple
import Random
using Test
using RectiGrids
using OnlineStatsBase
using StableRNGs
using StaticArrays
import StatsBase
using MultiCDFs


@testset "single point" begin
    rng = StableRNG(123)
    n = 100
    data_t2 = tuple.(randn(rng, n), randn(rng, n))
    data_sv2 = SVector.(data_t2)
    data_nt2 = NamedTuple{(:a, :b)}.(data_t2)
    data_r = getindex.(data_t2, 1)

    @testset "1d" begin
        @testset "numbers" begin
            ecdf = ECDF(data_r)
            sb_ecdf = StatsBase.ecdf(data_r)

            @test ecdf.(-2:0.5:2) ≈ sb_ecdf(-2:0.5:2)
            @test ecdf.(data_r) ≈ sb_ecdf(data_r)
            xs = randn(10)
            @test ecdf.(xs) ≈ sb_ecdf(xs)
            @test ecdf(-Inf) == 0
            @test ecdf(Inf) == 1

            @test_throws MethodError ecdf((0.,))
            @test_throws MethodError ecdf(SVector(0.,))
            
            @testset "orders" begin
                @test ecdf(-Inf, Orders.Below()) == 0
                @test ecdf(-Inf, Orders.Above()) == 1
                @test ecdf(Inf, Orders.Below()) == 1
                @test ecdf(Inf, Orders.Above()) == 0
                @test ecdf(0.) == ecdf(0., Orders.Below()) == 1 - ecdf(0., Orders.Above())
            end
        end

        @testset "tuples" begin
            ecdf = ECDF(tuple.(data_r))
            sb_ecdf = StatsBase.ecdf(data_r)
            @test ecdf.(tuple.(-2:0.5:2)) ≈ sb_ecdf(-2:0.5:2)
            @test_throws MethodError ecdf(0.)
            @test_throws MethodError ecdf(SVector(0.))
            @test_throws MethodError ecdf((a=0.,))
        end

        @testset "namedtuples" begin
            ecdf = ECDF(NamedTuple{(:a,)}.(tuple.(data_r)))
            sb_ecdf = StatsBase.ecdf(data_r)
            @test ecdf.(NamedTuple{(:a,)}.(tuple.(-2:0.5:2))) ≈ sb_ecdf(-2:0.5:2)
            @test ecdf((a=Inf,)) == 1
            @test_throws MethodError ecdf(0.)
            @test_throws MethodError ecdf((0.,))
            @test_throws MethodError ecdf(SVector(0.))
            @test_throws MethodError ecdf((b=0.,))
        end

        @testset "svectors" begin
            ecdf = ECDF(SVector.(data_r))
            sb_ecdf = StatsBase.ecdf(data_r)
            @test ecdf.(SVector.(-2:0.5:2)) ≈ sb_ecdf(-2:0.5:2)
            @test_throws MethodError ecdf(0.)
            @test_throws MethodError ecdf((0.,))
            @test_throws MethodError ecdf((a=0.,))
        end
    end

    @testset "2d" begin
        sb_ecdf1 = StatsBase.ecdf(getindex.(data_t2, 1))
        sb_ecdf2 = StatsBase.ecdf(getindex.(data_t2, 2))

        @testset "tuples" begin
            ecdf = ECDF(data_t2)
            @test ecdf.(tuple.(-2:0.5:2, Inf)) ≈ sb_ecdf1(-2:0.5:2)
            @test ecdf.(tuple.(Inf, -2:0.5:2)) ≈ sb_ecdf2(-2:0.5:2)
            @test ecdf((-Inf, -Inf)) == ecdf((0., -Inf)) == ecdf((Inf, -Inf)) == ecdf((-Inf, 0.)) == ecdf((-Inf, Inf)) == 0
            @test ecdf((Inf, Inf)) == 1
        end

        @testset "namedtuples" begin
            ecdf = ECDF(data_nt2)
            @test ecdf.(NamedTuple{(:a, :b)}.(tuple.(-2:0.5:2, Inf))) ≈ sb_ecdf1(-2:0.5:2)
            @test ecdf.(NamedTuple{(:a, :b)}.(tuple.(Inf, -2:0.5:2))) ≈ sb_ecdf2(-2:0.5:2)
            @test ecdf((a=-Inf, b=-Inf)) == ecdf((a=0., b=-Inf)) == ecdf((a=Inf, b=-Inf)) == ecdf((a=-Inf, b=0.)) == ecdf((a=-Inf, b=Inf)) == 0
            @test ecdf((a=Inf, b=Inf)) == 1
            @test_throws MethodError ecdf((b=0., a=0.))
            @test_throws MethodError ecdf((a=0., c=0.))
            
            @testset "orders" begin
                @test ecdf((a=-Inf, b=-Inf), (a=Orders.Below(), b=Orders.Below())) == 0
                @test_throws MethodError ecdf((a=-Inf, b=-Inf), (b=Orders.Below(), a=Orders.Below()))
                @test ecdf((a=-Inf, b=-Inf), (a=Orders.Above(), b=Orders.Above())) == 1
                @test ecdf((a=-Inf, b=-Inf), (a=Orders.Above(), b=Orders.Below())) == 0
                @test ecdf((a=0., b=0.), (a=Orders.Below(), b=Orders.Below())) +
                        ecdf((a=0., b=0.), (a=Orders.Above(), b=Orders.Above())) +
                        ecdf((a=0., b=0.), (a=Orders.Above(), b=Orders.Below())) +
                        ecdf((a=0., b=0.), (a=Orders.Below(), b=Orders.Above())) == 1
            end
        end
    end
end

@testset "grid" begin
    rng = StableRNG(123)
    n = 100
    data_t = tuple.(randn(rng, n), randn(rng, n))
    data_nt = NamedTuple{(:a, :b)}.(data_t)
    
    e1 = ecdf_evaluate(data_t, grid(-1:0.5:1, -1:1:2))
    @test e1 == [2 6 9 13; 4 12 22 28; 7 21 38 46; 9 27 50 58; 13 40 70 79]
    @test e1[1, 3] == 9
    @test e1(0, -1) == 7
    @test ecdf_evaluate(data_t, grid(-1:0.5:1, -1:1:2), aggregate=(Orders.Below(), Orders.Below())) == e1
    
    e2 = ecdf_evaluate(data_t, grid(-1:0.5:1, -1:1:2), aggregate=(Orders.Above(), Orders.Below()))
    @test e2 == [15 47 78 85; 13 41 65 70; 10 32 49 52; 8 26 37 40; 4 13 17 19]

    e3 = ecdf_evaluate(data_nt, grid(a=-1:0.5:1, b=-1:1:2))
    @test e3 == e1
    @test e3[1, 3] == 9
    @test e3(a=0, b=-1) == 7
    @test e3(b=-1, a=0) == 7

    e4 = ecdf_evaluate(data_nt, grid(a=-1:0.5:1, b=-1:1:2), aggregate=(a=Orders.Above(),))
    @test e4 == e2

    e5 = ecdf_evaluate(data_nt, grid(b=-1:1:2, a=-1:0.5:1), aggregate=(a=Orders.Above(),))
    @test e5 == permutedims(e4)

    e6 = ecdf_evaluate(Counter(NamedTuple), data_nt, grid(a=-1:0.5:1, b=-1:1:2))
    @test value.(e6) == e1

    e7 = ecdf_evaluate(FTSeries(NamedTuple, Counter(), Variance(), Mean(), transform=x -> x.a), data_nt, grid(a=-1:0.5:1, b=-1:1:2), aggregate=(a=Orders.NoAggBelow(),))
    @test value.(e7) == [(2, 0.027699667457727728, -1.3742978060979365) (6, 0.5438750517689237, -1.5802601596545798) (9, 0.38190063142303354, -1.5040656471466063) (13, 0.27805282947152743, -1.4110603294419615); (2, 0.037526726131552855, -0.8302181031058844) (6, 0.011058917594755018, -0.7639272361221531) (13, 0.016372968761058593, -0.6669982834402005) (15, 0.023627158117447308, -0.6784867854023856); (3, 0.023994683864457987, -0.27189392820009833) (9, 0.011272447125220634, -0.28184447895351034) (16, 0.01336414974584977, -0.2531332595829061) (18, 0.013641722223708578, -0.2665305246580923); (2, 0.0002854270126869635, 0.16981029914503348) (6, 0.028279124364474086, 0.22324241859197247) (12, 0.02425111677857292, 0.21919989329471354) (12, 0.02425111677857292, 0.21919989329471354); (4, 0.029413968789774737, 0.7570542383527122) (13, 0.021841556311418524, 0.7760191280392775) (20, 0.02559495975081888, 0.7519055191102784) (21, 0.02519187651769472, 0.7454444092453625)]
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
