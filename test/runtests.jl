import Random
using Test
using LazyGrids
using OnlineStatsBase
using MultiCDFs

@testset begin
    Random.seed!(123)
    n = 100
    data_t = tuple.(randn(n), randn(n))
    data_nt = NamedTuple{(:a, :b)}.(data_t)
    
    e1 = ecdf_evaluate(data_t, grid(-1:0.5:1, -1:1:2))
    @test e1 == [1 1 10 11; 4 9 28 30; 8 19 45 51; 10 26 60 68; 13 32 68 76]
    @test e1[1, 3] == 10
    @test e1(0, -1) == 8
    @test ecdf_evaluate(data_t, grid(-1:0.5:1, -1:1:2), aggregate=(Orders.Below(), Orders.Below())) == e1
    
    e2 = ecdf_evaluate(data_t, grid(-1:0.5:1, -1:1:2), aggregate=(Orders.Above(), Orders.Below()))
    @test e2 == [16 41 73 84; 13 33 55 65; 9 23 38 44; 7 16 23 27; 4 10 15 19]

    e3 = ecdf_evaluate(data_nt, grid(a=-1:0.5:1, b=-1:1:2))
    @test e3 == e1
    @test e3[1, 3] == 10
    @test e3(a=0, b=-1) == 8
    @test e3(b=-1, a=0) == 8

    e4 = ecdf_evaluate(data_nt, grid(a=-1:0.5:1, b=-1:1:2), aggregate=(a=Orders.Above(),))
    @test e4 == e2

    e5 = ecdf_evaluate(data_nt, grid(b=-1:1:2, a=-1:0.5:1), aggregate=(a=Orders.Above(),))
    @test e5 == permutedims(e4)

    e6 = ecdf_evaluate(Counter(NamedTuple), data_nt, grid(a=-1:0.5:1, b=-1:1:2))
    @test value.(e6) == e1

    e7 = ecdf_evaluate(FTSeries(NamedTuple, Counter(), Variance(), Mean(), transform=x -> x.a), data_nt, grid(a=-1:0.5:1, b=-1:1:2), aggregate=(a=Orders.NoAggBelow(),))
    @test value.(e7) == [(1, 1.0, -1.03343204404106) (1, 1.0, -1.03343204404106) (10, 0.5373640410461016, -1.7620478606519892) (11, 0.6082883896618717, -1.8685034650223216); (3, 0.01395605960438727, -0.7469596856637633) (8, 0.03359405125353527, -0.7622441779792377) (18, 0.02158134219693389, -0.7981066874746774) (19, 0.02072511309388542, -0.7938594942693962); (4, 0.02584917782892268, -0.17451310731345465) (10, 0.012241418504158934, -0.18707874587373557) (17, 0.01199058449905356, -0.23391008957798712) (21, 0.012881057531022536, -0.24422520010138682); (2, 0.001518157673196386, 0.24626398582778403) (7, 0.014093371291231914, 0.2136775537983112) (15, 0.015103695412707879, 0.28402384656190655) (17, 0.015730470978850934, 0.3014586737937683); (3, 0.017462846991453656, 0.7023758253333133) (6, 0.009317705509081326, 0.7384257276813919) (8, 0.014447569302655396, 0.7816536408445827) (8, 0.014447569302655396, 0.7816536408445827)]
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
