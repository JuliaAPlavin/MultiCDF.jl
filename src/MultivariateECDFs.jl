module MultivariateECDFs

export ecdf_evaluate, Orders

using Parameters
using LazyGrids
using OnlineStatsBase: OnlineStat, fit!, merge


module Orders
export Order, Below, Above, NoAggBelow, NoAggAbove
abstract type Order end
struct NoAggBelow <: Order end
struct NoAggAbove <: Order end
struct Below <: Order end
struct Above <: Order end
end
using .Orders

@generated select(nt::NamedTuple, _::NamedTuple{Kix}) where {Kix} = :( (;$([:(nt.$k) for k in Kix]...)) )


@with_kw struct ECDFAxisSpec{TV}
    binedges::TV
    order::Order
    
    @assert issorted(binedges)
end

Base.length(a::ECDFAxisSpec) = length(a.binedges)
findbin(a::ECDFAxisSpec, x) = findbin(a.order, a.binedges, x)

findbin(as::Tuple, xs::Union{Tuple, AbstractVector}) = map((ax, x) -> findbin(ax, x), as, xs)
findbin(as::NamedTuple, xs::NamedTuple) = map((ax, x) -> findbin(ax, x), as, select(xs, as)) |> values

findbin(::Union{Below, NoAggBelow}, binedges::AbstractVector, x) = searchsortedfirst(binedges, x)
findbin(::Union{Above, NoAggAbove}, binedges::AbstractVector, x) = searchsortedlast(binedges, x)

aggregate_axis!(::Union{NoAggBelow, NoAggAbove}, A::AbstractArray, dim::Int) = A
aggregate_axis!(::Below, A::AbstractArray{<:Number}, dim::Int) = cumsum!(A, A, dims=dim)
aggregate_axis!(::Below, A::AbstractArray, dim::Int) = accumulate!(merge, A, A, dims=dim)
aggregate_axis!(::Above, A::AbstractArray, dim::Int) = (reverse!(A, dims=dim); cumsum!(A, A, dims=dim); reverse!(A, dims=dim))


function ecdf_evaluate(data::AbstractVector{<:Tuple}, g::Grid; aggregate::NTuple{N, Order}=ntuple(_ -> Orders.Below(), ndims(g))) where {N}
	@assert eltype(g) <: Tuple
	axspecs = map(axiskeys(g), aggregate) do ax, agg
		ECDFAxisSpec(ax, agg)
	end
	counts = map(_ -> 0, KeyedArray(g))
	for r in data
		ix = findbin(axspecs, r) |> CartesianIndex
		checkbounds(Bool, counts, ix) || continue
		counts[ix] += 1
	end
	for (dim, ax) in enumerate(axspecs)
		aggregate_axis!(ax.order, counts, dim)
	end
	return counts
end

function ecdf_evaluate(data::AbstractVector{<:NamedTuple}, g::Grid; aggregate::NamedTuple=(;))
	@assert eltype(g) <: NamedTuple
	aggregate_ = merge(map(_ -> Orders.Below(), named_axiskeys(g)), aggregate)
	axspecs = map(named_axiskeys(g), aggregate_) do ax, agg
		ECDFAxisSpec(ax, agg)
	end
	counts = map(_ -> 0, KeyedArray(g))
	for r in data
		ix = findbin(axspecs, r) |> CartesianIndex
		checkbounds(Bool, counts, ix) || continue
		counts[ix] += 1
	end
	for (dim, ax) in enumerate(axspecs)
		aggregate_axis!(ax.order, counts, dim)
	end
	return counts
end

function ecdf_evaluate(stat::OnlineStat, data::AbstractVector{<:NamedTuple}, g::Grid; aggregate::NamedTuple=(;))
	@assert eltype(g) <: NamedTuple
	aggregate_ = merge(map(_ -> Orders.Below(), named_axiskeys(g)), aggregate)
	axspecs = map(named_axiskeys(g), aggregate_) do ax, agg
		ECDFAxisSpec(ax, agg)
	end
	result = map(_ -> copy(stat), KeyedArray(g))
	for r in data
		ix = findbin(axspecs, r) |> CartesianIndex
		checkbounds(Bool, result, ix) || continue
		fit!(result[ix], r)
	end
	for (dim, ax) in enumerate(axspecs)
		aggregate_axis!(ax.order, result, dim)
	end
	return result
end


end
