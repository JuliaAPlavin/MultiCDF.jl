module MultiCDFs

export ecdf_evaluate, Orders, ECDF

using Parameters
using RectiGrids
using OnlineStatsBase: OnlineStat, fit!, merge, value
using AxisKeys: hasnames


module Orders
export Order, Below, Above, NoAggBelow, NoAggAbove
abstract type Order end
struct NoAggBelow <: Order end
struct NoAggAbove <: Order end
struct Below <: Order end
struct Above <: Order end
end
using .Orders


struct ECDF{T, TD <: AbstractVector{T}}
	data::TD
end

function (ecdf::ECDF)(x)
	sum(y -> is_all_leq(y, x), ecdf.data) / length(ecdf.data)
end

function (ecdf::ECDF)(x, ::typeof(count))
	sum(y -> is_all_leq(y, x), ecdf.data)
end

function (ecdf::ECDF)(x, stat::OnlineStat)
	fit!(copy(stat), filter(y -> is_all_leq(y, x), ecdf.data))
end

Broadcast.broadcasted(ecdf::ECDF, g::RectiGrid) = ecdf.(g, count) ./ length(ecdf.data)

function Broadcast.broadcasted(ecdf::ECDF, g::RectiGrid, ::typeof(count))
	# aggregate_ = merge(map(_ -> Orders.Below(), named_axiskeys(g)), aggregate)
	axspecs = map(hasnames(g) ? named_axiskeys(g) : axiskeys(g)) do ax
		ECDFAxisSpec(ax, Orders.Below())
	end
	counts = map(_ -> 0, g)
	for r in ecdf.data
		ix = findbin(axspecs, r) |> CartesianIndex
		checkbounds(Bool, counts, ix) || continue
		counts[ix] += 1
	end
	for (dim, ax) in enumerate(axspecs)
		aggregate_axis!(ax.order, counts, dim)
	end
	return counts
end

function Broadcast.broadcasted(ecdf::ECDF, g::RectiGrid, stat::OnlineStat)
	axspecs = map(hasnames(g) ? named_axiskeys(g) : axiskeys(g)) do ax
		ECDFAxisSpec(ax, Orders.Below())
	end
	result = map(_ -> copy(stat), g)
	for r in ecdf.data
		ix = findbin(axspecs, r) |> CartesianIndex
		checkbounds(Bool, result, ix) || continue
		fit!(result[ix], r)
	end
	for (dim, ax) in enumerate(axspecs)
		aggregate_axis!(ax.order, result, dim)
	end
	return result
end

# like Base.AbstractVecOrTuple, but includes heterogeneous tuples
const AbstractVecOrTuple = Union{AbstractVector, Tuple}
is_all_leq(datap::Real, query::Real) = datap <= query
is_all_leq(datap::Real, query::Base.Fix2) = query(datap)
is_all_leq(datap::AbstractVecOrTuple, query::AbstractVecOrTuple) = (@assert length(datap) == length(query); all(is_all_leq.(datap, query)))
is_all_leq(datap::NamedTuple{NSD}, query::NamedTuple{NSQ}) where {NSD, NSQ} = is_all_leq(values(datap[NSQ]), values(query))

# is_all_leq(a::T, b::T, ::Below) where {T <: Real} = a <= b
# is_all_leq(a::T, b::T, ::Above) where {T <: Real} = a >= b
# is_all_leq(a::T, b::T, orders::Tuple) where {T <: Tuple} = all(is_all_leq.(a, b, orders))
# is_all_leq(a::T, b::T, orders::AbstractVector) where {T <: AbstractVector} = all(is_all_leq.(a, b, orders))
# is_all_leq(a::T, b::T, orders::NamedTuple{NS}) where {NS, T <: NamedTuple{NS}} = is_all_leq(values(a), values(b), values(orders))


@with_kw struct ECDFAxisSpec{TV, TO}
    binedges::TV
    order::TO
    
    @assert issorted(binedges)
end

@inline findbin(a::ECDFAxisSpec, x) = findbin(a.order, a.binedges, x)

@inline findbin(as::Tuple, xs::Union{Tuple, AbstractVector}) = map((ax, x) -> findbin(ax, x), as, xs)
@inline findbin(as::NamedTuple{NS}, xs::NamedTuple) where {NS} = map((ax, x) -> findbin(ax, x), as, xs[NS]) |> values

@inline findbin(::Union{Below, NoAggBelow}, binedges::AbstractVector, x) = searchsortedfirst(binedges, x)
@inline findbin(::Union{Above, NoAggAbove}, binedges::AbstractVector, x) = searchsortedlast(binedges, x)

aggregate_axis!(::Union{NoAggBelow, NoAggAbove}, A::AbstractArray, dim::Int) = A
aggregate_axis!(::Below, A::AbstractArray{<:Number}, dim::Int) = cumsum!(A, A, dims=dim)
aggregate_axis!(::Below, A::AbstractArray, dim::Int) = accumulate!(merge, A, A, dims=dim)
aggregate_axis!(::Above, A::AbstractArray, dim::Int) = (reverse!(A, dims=dim); cumsum!(A, A, dims=dim); reverse!(A, dims=dim))


end
