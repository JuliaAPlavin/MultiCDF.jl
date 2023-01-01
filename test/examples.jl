### A Pluto.jl notebook ###
# v0.17.5

using Markdown
using InteractiveUtils

# ╔═╡ 5d7ddd57-2b80-430e-8b1a-a6dc72ae31b9
begin
	using Revise
	# import Pkg
	# eval(:(Pkg.develop(path="..")))
	using MultiCDF
end

# ╔═╡ c1c7165e-9f29-4d6b-b8e0-125dbb8cfbd6
using BenchmarkTools

# ╔═╡ 5aca4551-b811-4990-862a-193e2815c07a
using Altair

# ╔═╡ c83704d4-5542-48bf-a561-281e42b8836d
using RectiGrids

# ╔═╡ 4ec1d7d7-f52f-41f4-abc6-e48c44a7a8d8
using DisplayAs: Text as AsText

# ╔═╡ f8de31e5-e9a0-4a9b-b979-e6bae334eb44
using OnlineStatsBase

# ╔═╡ 42535172-bd70-42fe-9792-80b12d357806
using DataPipes

# ╔═╡ 65ee6e67-3de2-4ad1-8bf2-775f75e29312
using Accessors

# ╔═╡ 1e5164c2-e967-4a25-84e6-66f6436fe794
md"""
Load required packages:
"""

# ╔═╡ 33ef67cf-6da4-4867-a41c-b082674efd81
md"""
## Create `ECDF`
"""

# ╔═╡ b2521d9c-3ebc-4ebe-a0e2-74de82ed3a35
md"""
Generate some 2d data as `Tuple`s and `NamedTuple`s:
"""

# ╔═╡ 316c6a8a-c39b-4fc2-a647-de46c6ed9766
n = 100

# ╔═╡ 85506171-9280-4f4a-97ad-8249faf005a2
data_t2 = Tuple.(eachrow(randn(n, 2)))

# ╔═╡ 563a6141-fc8a-4ec0-b6ff-225f8e1efafb
data_nt2 = NamedTuple{(:a, :b)}.(data_t2)

# ╔═╡ e675c56e-7fb6-44a6-adaf-91c932a2e1c6
md"""
Create `ECDF` instances for these datasets. For now, `ECDF` just stores the data array without any processing.
"""

# ╔═╡ ac9ed157-99a1-46e2-907d-3c64f7fbe939
ecdf_t = ecdf(data_t2)

# ╔═╡ 3f00bb9a-49e4-4b04-8177-d51931bc8e7f
ecdf_nt = ecdf(data_nt2)

# ╔═╡ 465f4319-6f7f-4d56-b8cf-df7aceffc956
md"""
The `ECDF.signs` field stores comparison signs for generalized CDFs. See below for examples.
"""

# ╔═╡ da68a0f4-997e-4a05-8107-872ba3369f54
ecdf_nt.signs

# ╔═╡ a0d27d5a-6f9f-46c8-966c-e3eaf713e081
md"""
## Simple ECDF computations
"""

# ╔═╡ d08d78bd-a645-42e7-b909-f9ae9f81f5e4
md"""
Compute empirical CDF at a single point, that is $CDF(x) = \#\{y \in data : \forall i \ y_i \leq x_i\} \ /\ \# data$
"""

# ╔═╡ 041364da-9656-4312-bad6-4b62315a386a
ecdf_t((0, 0))

# ╔═╡ d97d7658-e49c-411c-801c-35491f7dadfb
ecdf_nt((a=0, b=0))

# ╔═╡ 668a6a1f-5408-435c-8c28-e88f724202cf
md"""
Genetalized CDF with arbitrary signs, like `<`, `>=`, ...:
"""

# ╔═╡ fe3a1de6-d1df-4656-8979-c9d721a0ce4e
ecdf_nt((a= <(0), b= <(0)))

# ╔═╡ 75464c77-5cd3-4a50-8af0-a5e1f20c8e4d
ecdf_nt((a= >(0), b= <=(Inf)))

# ╔═╡ 29362502-23ac-40ca-a8ac-2ac174d8fe5f
md"""
Pass `count` to compute integer counts instead of fractions:
"""

# ╔═╡ 0e380819-4dbf-49f9-9842-36c3ebca4227
ecdf_nt((a=0., b=0.), count)

# ╔═╡ 123577d3-8599-4968-a68f-d1b257723f5e
ecdf_nt((a=0., b=<=(Inf)), count)

# ╔═╡ 6b87d8d8-4b7a-441d-adf1-3eef4dea0245
md"""
Lower-dimensional marginal ECDFs can also be computed, down to zero dims:
"""

# ╔═╡ 7c106770-3c09-432b-a397-7f811b81da25
ecdf_nt((a=0,))

# ╔═╡ 80356436-1fd2-4941-a0b1-9144d0b4519a
ecdf_nt((a= >(-Inf),))

# ╔═╡ 384acb45-b4cb-4323-b837-f27ad3ca834c
ecdf_nt((;), count)

# ╔═╡ dc915af1-076f-4cb7-be2e-785988e3202b
ecdf(NamedTuple{(:a,)}[])((a=1,), count)

# ╔═╡ bad35ee4-4142-4d27-b2c9-e9490d807cfc
md"""
This only works for `NamedTuple`s, not regular `Tuple`s.
"""

# ╔═╡ c467316a-f248-4968-be51-c3b932c05af6
md"""
## ECDF on a grid
"""

# ╔═╡ 0239e2b1-c070-460c-ba3e-9bd8e7854a83
md"""
Computing CDF at $M$ points the naive way would be suboptimal and require $O(M \cdot N)$ operations, where $N = \# data$.

However, for target points on a regular grid, a faster $O(N \log M + M)$ solution is possible. It becomes $O(N + M)$ when grid axes are `range`s and not arbitrary `Array`s.

The optimized algorithm is invoked when `ECDF` is broadcasted over a `grid` from the `RectiGrids` package:
"""

# ╔═╡ 493795e3-70d6-43fb-b9ae-5c2ec699642d
ecdf_nt.(grid(a=-2:2, b=-2:2)) |> AsText

# ╔═╡ 3f887f2d-8567-4e36-b224-1a1a38b1e71e
ecdf_nt.(grid(a=-2:2, b=-2:2), count) |> AsText

# ╔═╡ 5749b7f4-c418-483e-b1d7-ec6ab86a2113
md"""
Resulting values are the same as with a naive approach, of course:
"""

# ╔═╡ 0d6cad73-94f4-4deb-bc5d-3ebe72cb115c
ecdf_nt.(collect(grid(a=-2:2, b=-2:2))) |> AsText

# ╔═╡ 5d941d00-f930-442c-ad18-b61a1da94219
md"""
Everything works with other signs as well:
"""

# ╔═╡ 64826fef-2683-4050-a0d5-8a7034a615bb
ecdf_nt_s = @set ecdf_nt.signs = (a= <=, b= >=)

# ╔═╡ 5972b8f9-9246-4f9a-b17c-bb655545a558
ecdf_nt_s.(grid(a=-2:2, b=-2:2), count) |> AsText

# ╔═╡ 2e342913-17c0-4a39-ae5a-3d0145629033
ecdf_nt_s.(collect(grid(a=-2:2, b=-2:2)), count) |> AsText

# ╔═╡ df6ef402-87a3-40de-9780-fad770833de5
md"""
Marginal ECDF on a grid:
"""

# ╔═╡ 4ae9cacd-45c9-4392-bbb9-3336d1bff9e6
ecdf_nt.(grid(a=-2:2), count) |> AsText

# ╔═╡ 3b4ddd22-c6a8-46a5-b098-b1ce0ba7cbdf
md"""
## `OnlineStats` integration
"""

# ╔═╡ 0b60da02-1994-4c2c-aaf1-7cb5cc00aaaa
md"""
In addition to simple counting, any `OnlineStats` statistics can be computed for data points counted in the CDF: $points(x) = \{y \in data : \forall i \ y_i \leq x_i\}$.
"""

# ╔═╡ 533dcc45-a15c-489a-8ea1-6dfafe7fbacc
md"""
Using `Counter` gives the same result as plain counting:
"""

# ╔═╡ eecc58f8-c959-42b9-b375-67f22205c404
ecdf_nt.(grid(a=-2:2, b=-2:2), Counter(NamedTuple)) .|> value |> AsText

# ╔═╡ dff12e8b-9cee-4700-9f23-aeddbe69c6be
ecdf_nt.(grid(a=-2:2, b=-2:2), count) |> AsText

# ╔═╡ b6aa4f71-1699-4d9d-946b-4e6a4c2fab0d
md"""
Any other statistic can be used for more advanced computations. Eg, a componentwise mean:
"""

# ╔═╡ fd46712a-ce4e-4303-ae2c-95573acfd7f0
ecdf_nt.(grid(a=-2:2, b=-2:2), Group(Mean(), Mean())) .|> value |> AsText

# ╔═╡ fb10d8e1-efe9-49f2-9611-45971116221a
md"""
Weighted ECDFs can be computed this way.
"""

# ╔═╡ 0dd1aa81-2762-498d-9801-f7344d9db8c2
md"""
## Benchmarks
"""

# ╔═╡ 8bc27929-0c71-4424-9c50-0381dae7c712
md"""
All benchmarks are run with built-in counting and using `OnlineStats.Counter`. Grid axes are either vectors or ranges.

`ndata` is the number of datapoints, `ngrid` is the total number of grid points, `ndims` is the dimensionality of each point.

Note that the online HTML is generated with GitHub Actions, and timings can be slow or noisy.
"""

# ╔═╡ 1038d519-47f2-41e5-8d0a-ac44a5aafc71
md"""
Creating ECDF object. For now, it just takes the provided dataset as-is.
"""

# ╔═╡ 00c41289-debc-492e-950e-a4ce94e86af8
md"""
Computing ECDF at a single point. Timings for small `ndata` are influenced by Julia `@timed` overhead.
"""

# ╔═╡ 93c43a68-2651-47a0-896a-d590a61eef25
md"""
Computing ECDF at a grid of points. First and second rows differ in the horizontal axis and colorbar: `ndata` and `ngrid` are swapped.
"""

# ╔═╡ 47131891-60e9-4c51-95ae-c36dbdc6a2ba
md"""
Timing computations:
"""

# ╔═╡ 449c9cd1-87eb-4625-be85-947a632ab4cf
do_bench = true

# ╔═╡ 4af19b6a-204f-4efb-a91c-4388b576dd1b
timings = do_bench && map(grid(
		gridtype = [:range, :vector],
	cnttype = [:native, :onlinestats],
		ndims=1:3,
		ndata=10 .^ (1:6),
		ngrid=10 .^ (1:6),
	)) do g
	ks = (:x, :y, :z)[1:g.ndims]
	data = NamedTuple{ks}.(eachrow(randn(g.ndata, g.ndims)))
	t_cdf = @timed ecdf(data)
	cdf = t_cdf.value
	t1 = if g.cnttype == :native
		@timed cdf(data[1])
	elseif g.cnttype == :onlinestats
		@timed cdf(data[1], Counter(NamedTuple))
	end

	gridside = round(Int, g.ngrid^(1/g.ndims))
	G = grid(; (k => range(-2, 2, length=gridside) |> Dict(:range => identity, :vector => collect)[g.gridtype] for k in ks)...)
	tG = if g.cnttype == :native
		@timed cdf.(G)
	elseif g.cnttype == :onlinestats
		@timed cdf.(G, Counter(NamedTuple)) .|> value
	end

	(; g..., t_cdf=t_cdf[(:time, :bytes)], t1=t1[(:time, :bytes)], tG=tG[(:time, :bytes)])
end

# ╔═╡ e9d22825-4909-4230-bb30-375298d2ede9
let
	ch₀ = @p begin
		timings
		map(unnest)
		mutate(
			detail=string(_[(:ngrid, :cnttype, :gridtype)])
		)
		altChart()
	end
	ch₀ = ch₀.mark_line()
	ch_t1 = ch₀.encode(
		alt.X(:ndata, scale=alt.Scale(type=:log)),
		alt.Y(:t_cdf_time, scale=alt.Scale(type=:log, nice=false), axis=alt.Axis(format="~s")),
		alt.Detail(:detail),
		alt.Color(:ndims, type=:nominal),
	)
	altVLSpec(ch_t1)
end

# ╔═╡ 894b35d3-9e33-4455-9bf5-0ec505a47f90
let
	ch₀ = @p begin
		timings
		map(unnest)
		mutate(
			typ=string(_[(:cnttype, :gridtype)]),
			detail=string(_[(:ngrid, :gridtype)])
		)
		altChart()
	end
	ch₀ = ch₀.mark_line()
	ch_t1 = ch₀.encode(
		alt.X(:ndata, scale=alt.Scale(type=:log)),
		alt.Y(:t1_time, scale=alt.Scale(type=:log, nice=false), axis=alt.Axis(format="~s")),
		alt.Detail(:detail),
		alt.Color(:ndims, type=:nominal),
	).facet(:cnttype)
	altVLSpec(ch_t1)
end

# ╔═╡ bbc3e652-32a3-4dbd-9983-f1151a381497
let
	ch₀ = @p begin
		timings
		map(unnest)
		mutate(
			ndims_type="$(_.ndims) $(_.gridtype)",
			ngrid_type="$(_.ngrid) $(_.gridtype)",
			typ=string(_[(:cnttype, :gridtype)]),
		)
		altChart()
	end
	ch₀ = ch₀.mark_line()
	ch_tG₀ = ch₀.encode(
		alt.Y(:tG_time, scale=alt.Scale(type=:log, nice=false), axis=alt.Axis(format="~s")),
		alt.StrokeDash(:ndims, type=:ordinal),
	)
	chs_tG₀ = ch_tG₀.encode(
		alt.X(:ndata, scale=alt.Scale(type=:log)),
		alt.Color(:ngrid, scale=alt.Scale(type=:log, nice=false, scheme=:turbo)),
	).facet(:typ)

	ch = chs_tG₀
	altVLSpec(chs_tG₀)
end

# ╔═╡ 6112c390-ef14-4c7a-b811-64726359f7e7
let
	ch₀ = @p begin
		timings
		map(unnest)
		mutate(
			ndims_type="$(_.ndims) $(_.gridtype)",
			ngrid_type="$(_.ngrid) $(_.gridtype)",
			typ=string(_[(:cnttype, :gridtype)]),
		)
		altChart()
	end
	ch₀ = ch₀.mark_line()
	ch_tG₀ = ch₀.encode(
		alt.Y(:tG_time, scale=alt.Scale(type=:log, nice=false), axis=alt.Axis(format="~s")),
		alt.StrokeDash(:ndims, type=:ordinal),
	)
	chs_tG₀ = ch_tG₀.encode(
		alt.X(:ngrid, scale=alt.Scale(type=:log)),
		alt.Color(:ndata, scale=alt.Scale(type=:log, nice=false, scheme=:turbo)),
	).facet(:typ)

	ch = chs_tG₀
	altVLSpec(chs_tG₀)
end

# ╔═╡ 8a76aaec-d4d8-4d53-909f-c3410ccbf5b8
length(timings)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Accessors = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
Altair = "b5d8985d-ff0a-46fa-83e6-c6893fdbcf16"
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
DataPipes = "02685ad9-2d12-40c3-9f73-c6aeda6a7ff5"
DisplayAs = "0b91fe84-8a4c-11e9-3e1d-67c38462b6d6"
MultiCDF = "663b7897-4180-4011-967b-e4930277ef1a"
OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
RectiGrids = "8ac6971d-971d-971d-971d-971d5ab1a71a"
Revise = "295af30f-e4ad-537b-8983-00126c2a3abe"

[compat]
Accessors = "~0.1.7"
Altair = "~0.1.1"
BenchmarkTools = "~1.2.2"
DataPipes = "~0.2.4"
DisplayAs = "~0.1.2"
MultiCDF = "~0.1.0"
OnlineStatsBase = "~1.4.9"
RectiGrids = "~0.1.6"
Revise = "~3.3.1"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.1"
manifest_format = "2.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "485ee0867925449198280d4af84bdb46a2a404d0"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.0.1"

[[deps.AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[deps.Accessors]]
deps = ["Compat", "CompositionsBase", "ConstructionBase", "Future", "LinearAlgebra", "MacroTools", "Requires", "Test"]
git-tree-sha1 = "2e427a6196c7aad4ee35054a9a90e9cb5df5c607"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.7"

[[deps.Adapt]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9faf218ea18c51fcccaf956c8d39614c9d30fe8b"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.3.2"

[[deps.Altair]]
deps = ["JSON", "Pandas", "PyCall", "VegaLite"]
git-tree-sha1 = "46fbac54bc17f3f0a07a808451ab6814e0b49a4d"
uuid = "b5d8985d-ff0a-46fa-83e6-c6893fdbcf16"
version = "0.1.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[deps.ArrayInterface]]
deps = ["Compat", "IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "1ee88c4c76caa995a885dc2f22a5d548dfbbc0ba"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.2.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisKeys]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "IntervalSets", "InvertedIndices", "LazyStack", "LinearAlgebra", "NamedDims", "OffsetArrays", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "8fd46bee70b52bfc11d90f5db51b33bda9c148df"
uuid = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
version = "0.1.24"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "940001114a0147b6e4d10624276d56d531dd9b49"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "926870acb6cbcf029396f2f2de030282b6bc1941"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.11.4"

[[deps.ChangesOfVariables]]
deps = ["ChainRulesCore", "LinearAlgebra", "Test"]
git-tree-sha1 = "bf98fa45a0a4cee295de98d4c1462be26345b9a1"
uuid = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
version = "0.1.2"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "9aa8a5ebb6b5bf469a7e0e2b5202cf6f8c291104"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.0.6"

[[deps.Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "44c37b4636bc54afac5c574d2d02b625349d6582"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.41.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.CompositionsBase]]
git-tree-sha1 = "455419f7e328a1a2493cabc6428d79e951349769"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.1"

[[deps.Conda]]
deps = ["Downloads", "JSON", "VersionParsing"]
git-tree-sha1 = "6cdc8832ba11c7695f494c9d9a1c31e90959ce0f"
uuid = "8f4d0f93-b110-5947-807f-2305c1781a2d"
version = "1.6.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[deps.CovarianceEstimation]]
deps = ["LinearAlgebra", "Statistics", "StatsBase"]
git-tree-sha1 = "a3e070133acab996660d31dcf479ea42849e368f"
uuid = "587fd27a-f159-11e8-2dae-1979310e6154"
version = "0.2.7"

[[deps.DataAPI]]
git-tree-sha1 = "cc70b17275652eb47bc9e5f81635981f13cea5c8"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.9.0"

[[deps.DataPipes]]
deps = ["SplitApplyCombine"]
git-tree-sha1 = "d16443eab0f2a312cb71227e4da49817fdff44b6"
uuid = "02685ad9-2d12-40c3-9f73-c6aeda6a7ff5"
version = "0.2.4"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "3daef5523dd2e769dad2365274f760ff5f282c7d"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.11"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.DataValues]]
deps = ["DataValueInterfaces", "Dates"]
git-tree-sha1 = "d88a19299eba280a6d062e135a43f00323ae70bf"
uuid = "e7dc6d0d-1eca-5fa6-8ad6-5aecde8b7ea5"
version = "0.4.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[deps.Dictionaries]]
deps = ["Indexing", "Random"]
git-tree-sha1 = "66bde31636301f4d217a161cabe42536fa754ec8"
uuid = "85a47980-9c8c-11e8-2b9f-f7ca1fa99fb4"
version = "0.3.17"

[[deps.DisplayAs]]
git-tree-sha1 = "44e8d47bc0b56ec09115056a692e5fa0976bfbff"
uuid = "0b91fe84-8a4c-11e9-3e1d-67c38462b6d6"
version = "0.1.2"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "b19534d1895d702889b219c382a6e18010797f0b"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.6"

[[deps.Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[deps.EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "3fe985505b4b667e1ae303c9ca64d181f09d5c05"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.3"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "67551df041955cc6ee2ed098718c8fcd7fc7aebe"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.12.0"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "04d13bfa8ef11720c24e4d840c0033d145537df7"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.17"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.HTTP]]
deps = ["Base64", "Dates", "IniFile", "Logging", "MbedTLS", "NetworkOptions", "Sockets", "URIs"]
git-tree-sha1 = "0fa77022fe4b511826b39c894c90daf5fce3334a"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "0.9.17"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.Indexing]]
git-tree-sha1 = "ce1566720fd6b19ff3411404d4b977acd4814f9f"
uuid = "313cdc1a-70c2-5d6a-ae34-0150d3930a38"
version = "1.1.1"

[[deps.IniFile]]
deps = ["Test"]
git-tree-sha1 = "098e4d2c533924c921f9f9847274f2ad89e018b8"
uuid = "83e8ac13-25f8-5344-8a64-a9f2b223428f"
version = "0.5.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[deps.InverseFunctions]]
deps = ["Test"]
git-tree-sha1 = "a7254c0acd8e62f1ac75ad24d5db43f5f19f3c65"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.2"

[[deps.InvertedIndices]]
git-tree-sha1 = "bee5f1ef5bf65df56bdd2e40447590b272a5471f"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.1.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "7fd44fd4ff43fc60815f8e764c0f352b83c49151"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.1.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "8076680b162ada2a031f707ac7b4953e30667a37"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.2"

[[deps.JSONSchema]]
deps = ["HTTP", "JSON", "URIs"]
git-tree-sha1 = "2f49f7f86762a0fbbeef84912265a1ae61c4ef80"
uuid = "7d188eb4-7ad8-530c-ae41-71a32a6d4692"
version = "0.3.4"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "a2366b16704ffe78be1831341e6799ab2f4f07d2"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.0"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyStack]]
deps = ["LinearAlgebra", "NamedDims", "OffsetArrays", "Test", "ZygoteRules"]
git-tree-sha1 = "a8bf67afad3f1ee59d367267adb7c44ccac7fdee"
uuid = "1fad7336-0346-5a1a-a56f-a06ba010965b"
version = "0.0.7"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["ChainRulesCore", "ChangesOfVariables", "DocStringExtensions", "InverseFunctions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "e5718a00af0ab9756305a0392832c8952c7426c1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.6"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "f46e8f4e38882b32dcc11c8d31c131d556063f39"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "2.2.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "3d3e902b31198a27340d0bf00d6ac452866021cf"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.9"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "Random", "Sockets"]
git-tree-sha1 = "1c38e51c3d08ef2278062ebceade0e46cefc96fe"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.0.3"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "bf210ce90b6c9eed32d25dbcae1ebc565df2687f"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.2"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[deps.MultiCDF]]
deps = ["AxisKeys", "OnlineStatsBase", "RectiGrids"]
git-tree-sha1 = "838a03239c5b09f0e027c365777173285abbba79"
uuid = "663b7897-4180-4011-967b-e4930277ef1a"
version = "0.1.0"

[[deps.NamedDims]]
deps = ["AbstractFFTs", "ChainRulesCore", "CovarianceEstimation", "LinearAlgebra", "Pkg", "Requires", "Statistics"]
git-tree-sha1 = "88dce79529a358f6efd13225d131bec958a18f1d"
uuid = "356022a1-0364-5f58-8944-0da4b18d706f"
version = "0.2.43"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[deps.NodeJS]]
deps = ["Pkg"]
git-tree-sha1 = "905224bbdd4b555c69bb964514cfa387616f0d3a"
uuid = "2bd173c7-0d6d-553b-b6af-13a54713934c"
version = "1.3.0"

[[deps.OffsetArrays]]
deps = ["Adapt"]
git-tree-sha1 = "043017e0bdeff61cfbb7afeb558ab29536bbb5ed"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.10.8"

[[deps.OnlineStatsBase]]
deps = ["AbstractTrees", "Dates", "LinearAlgebra", "OrderedCollections", "Statistics", "StatsBase"]
git-tree-sha1 = "287bd0f7ee1cc2a73f08057a7a6fcfe0c23fe4b0"
uuid = "925886fa-5bf2-5e8e-b522-a9147a512338"
version = "1.4.9"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Pandas]]
deps = ["Compat", "DataValues", "Dates", "IteratorInterfaceExtensions", "Lazy", "OrderedCollections", "Pkg", "PyCall", "Statistics", "TableTraits", "TableTraitsUtils", "Tables"]
git-tree-sha1 = "beefaeb19a644d5166c7b2dff9084ee0e63934a0"
uuid = "eadc2687-ae89-51f9-a5d9-86b5a6373a9c"
version = "1.5.3"

[[deps.Parsers]]
deps = ["Dates"]
git-tree-sha1 = "d7fa6237da8004be601e19bd6666083056649918"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.1.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Profile]]
deps = ["Printf"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"

[[deps.PyCall]]
deps = ["Conda", "Dates", "Libdl", "LinearAlgebra", "MacroTools", "Serialization", "VersionParsing"]
git-tree-sha1 = "71fd4022ecd0c6d20180e23ff1b3e05a143959c2"
uuid = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
version = "1.93.0"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.RectiGrids]]
deps = ["AxisKeys", "Random"]
git-tree-sha1 = "f7e83e8dcc2b6e78d09331720d75cc37355d122f"
uuid = "8ac6971d-971d-971d-971d-971d5ab1a71a"
version = "0.1.6"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "8f82019e525f4d5c669692772a6f4b0a58b06a6a"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.2.0"

[[deps.Revise]]
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "Pkg", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "2f9d4d6679b5f0394c52731db3794166f49d5131"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "fca29e68c5062722b5b4435594c3d1ba557072a3"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SplitApplyCombine]]
deps = ["Dictionaries", "Indexing"]
git-tree-sha1 = "dec0812af1547a54105b4a6615f341377da92de6"
uuid = "03a91e81-4c3e-53e1-a0a4-9c0c8f19dd66"
version = "1.2.0"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "7f5a513baec6f122401abfc8e9c074fdac54f6c1"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.4.1"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StatsAPI]]
git-tree-sha1 = "d88665adc9bcf45903013af0982e2fd05ae3d0a6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.2.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "51383f2d367eb3b444c961d485c565e4c0cf4ba0"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.14"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.TableTraitsUtils]]
deps = ["DataValues", "IteratorInterfaceExtensions", "Missings", "TableTraits"]
git-tree-sha1 = "78fecfe140d7abb480b53a44f3f85b6aa373c293"
uuid = "382cd787-c1b6-5bf2-a167-d5b971a19bda"
version = "1.0.2"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "bb1064c9a84c52e277f1096cf41434b675cd368b"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.6.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.URIParser]]
deps = ["Unicode"]
git-tree-sha1 = "53a9f49546b8d2dd2e688d216421d050c9a31d0d"
uuid = "30578b45-9adc-5946-b283-645ec420af67"
version = "0.4.1"

[[deps.URIs]]
git-tree-sha1 = "97bbe755a53fe859669cd907f2d96aee8d2c1355"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.3.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.Vega]]
deps = ["DataStructures", "DataValues", "Dates", "FileIO", "FilePaths", "IteratorInterfaceExtensions", "JSON", "JSONSchema", "MacroTools", "NodeJS", "Pkg", "REPL", "Random", "Setfield", "TableTraits", "TableTraitsUtils", "URIParser"]
git-tree-sha1 = "43f83d3119a868874d18da6bca0f4b5b6aae53f7"
uuid = "239c3e63-733f-47ad-beb7-a12fde22c578"
version = "2.3.0"

[[deps.VegaLite]]
deps = ["Base64", "DataStructures", "DataValues", "Dates", "FileIO", "FilePaths", "IteratorInterfaceExtensions", "JSON", "MacroTools", "NodeJS", "Pkg", "REPL", "Random", "TableTraits", "TableTraitsUtils", "URIParser", "Vega"]
git-tree-sha1 = "3e23f28af36da21bfb4acef08b144f92ad205660"
uuid = "112f6efa-9a02-5b7d-90c0-432ed331239a"
version = "2.6.0"

[[deps.VersionParsing]]
git-tree-sha1 = "e575cf85535c7c3292b4d89d89cc29e8c3098e47"
uuid = "81def892-9a0e-5fdd-b105-ffc91e053289"
version = "1.2.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[deps.ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "8c1a8e4dfacb1fd631745552c8db35d0deb09ea0"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─1e5164c2-e967-4a25-84e6-66f6436fe794
# ╠═5d7ddd57-2b80-430e-8b1a-a6dc72ae31b9
# ╠═c1c7165e-9f29-4d6b-b8e0-125dbb8cfbd6
# ╠═5aca4551-b811-4990-862a-193e2815c07a
# ╠═c83704d4-5542-48bf-a561-281e42b8836d
# ╠═4ec1d7d7-f52f-41f4-abc6-e48c44a7a8d8
# ╠═f8de31e5-e9a0-4a9b-b979-e6bae334eb44
# ╠═42535172-bd70-42fe-9792-80b12d357806
# ╠═65ee6e67-3de2-4ad1-8bf2-775f75e29312
# ╟─33ef67cf-6da4-4867-a41c-b082674efd81
# ╟─b2521d9c-3ebc-4ebe-a0e2-74de82ed3a35
# ╟─316c6a8a-c39b-4fc2-a647-de46c6ed9766
# ╠═85506171-9280-4f4a-97ad-8249faf005a2
# ╠═563a6141-fc8a-4ec0-b6ff-225f8e1efafb
# ╟─e675c56e-7fb6-44a6-adaf-91c932a2e1c6
# ╠═ac9ed157-99a1-46e2-907d-3c64f7fbe939
# ╠═3f00bb9a-49e4-4b04-8177-d51931bc8e7f
# ╟─465f4319-6f7f-4d56-b8cf-df7aceffc956
# ╠═da68a0f4-997e-4a05-8107-872ba3369f54
# ╟─a0d27d5a-6f9f-46c8-966c-e3eaf713e081
# ╟─d08d78bd-a645-42e7-b909-f9ae9f81f5e4
# ╠═041364da-9656-4312-bad6-4b62315a386a
# ╠═d97d7658-e49c-411c-801c-35491f7dadfb
# ╟─668a6a1f-5408-435c-8c28-e88f724202cf
# ╠═fe3a1de6-d1df-4656-8979-c9d721a0ce4e
# ╠═75464c77-5cd3-4a50-8af0-a5e1f20c8e4d
# ╟─29362502-23ac-40ca-a8ac-2ac174d8fe5f
# ╠═0e380819-4dbf-49f9-9842-36c3ebca4227
# ╠═123577d3-8599-4968-a68f-d1b257723f5e
# ╟─6b87d8d8-4b7a-441d-adf1-3eef4dea0245
# ╠═7c106770-3c09-432b-a397-7f811b81da25
# ╠═80356436-1fd2-4941-a0b1-9144d0b4519a
# ╠═384acb45-b4cb-4323-b837-f27ad3ca834c
# ╠═dc915af1-076f-4cb7-be2e-785988e3202b
# ╟─bad35ee4-4142-4d27-b2c9-e9490d807cfc
# ╟─c467316a-f248-4968-be51-c3b932c05af6
# ╟─0239e2b1-c070-460c-ba3e-9bd8e7854a83
# ╠═493795e3-70d6-43fb-b9ae-5c2ec699642d
# ╠═3f887f2d-8567-4e36-b224-1a1a38b1e71e
# ╟─5749b7f4-c418-483e-b1d7-ec6ab86a2113
# ╠═0d6cad73-94f4-4deb-bc5d-3ebe72cb115c
# ╟─5d941d00-f930-442c-ad18-b61a1da94219
# ╠═64826fef-2683-4050-a0d5-8a7034a615bb
# ╠═5972b8f9-9246-4f9a-b17c-bb655545a558
# ╠═2e342913-17c0-4a39-ae5a-3d0145629033
# ╟─df6ef402-87a3-40de-9780-fad770833de5
# ╠═4ae9cacd-45c9-4392-bbb9-3336d1bff9e6
# ╟─3b4ddd22-c6a8-46a5-b098-b1ce0ba7cbdf
# ╟─0b60da02-1994-4c2c-aaf1-7cb5cc00aaaa
# ╟─533dcc45-a15c-489a-8ea1-6dfafe7fbacc
# ╠═eecc58f8-c959-42b9-b375-67f22205c404
# ╠═dff12e8b-9cee-4700-9f23-aeddbe69c6be
# ╟─b6aa4f71-1699-4d9d-946b-4e6a4c2fab0d
# ╠═fd46712a-ce4e-4303-ae2c-95573acfd7f0
# ╟─fb10d8e1-efe9-49f2-9611-45971116221a
# ╟─0dd1aa81-2762-498d-9801-f7344d9db8c2
# ╟─8bc27929-0c71-4424-9c50-0381dae7c712
# ╟─1038d519-47f2-41e5-8d0a-ac44a5aafc71
# ╟─e9d22825-4909-4230-bb30-375298d2ede9
# ╟─00c41289-debc-492e-950e-a4ce94e86af8
# ╟─894b35d3-9e33-4455-9bf5-0ec505a47f90
# ╟─93c43a68-2651-47a0-896a-d590a61eef25
# ╟─bbc3e652-32a3-4dbd-9983-f1151a381497
# ╟─6112c390-ef14-4c7a-b811-64726359f7e7
# ╟─47131891-60e9-4c51-95ae-c36dbdc6a2ba
# ╠═449c9cd1-87eb-4625-be85-947a632ab4cf
# ╠═8a76aaec-d4d8-4d53-909f-c3410ccbf5b8
# ╟─4af19b6a-204f-4efb-a91c-4388b576dd1b
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
