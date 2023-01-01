module MultiCDFs

export ecdf

using RectiGrids
using OnlineStatsBase: OnlineStat, fit!, merge, value
using AxisKeys: hasnames


struct ECDF{T, TD <: AbstractVector{T}, TS}
	data::TD
	signs::TS
end

default_signs(::Type{<:NamedTuple{NS}}) where {NS} = NamedTuple{NS}(ntuple(_ -> <=, length(NS)))
default_signs(::Type{T}) where {T <: Tuple} = ntuple(_ -> <=, fieldcount(T))
default_signs(::Type{T}) where {T <: AbstractVector} = ntuple(_ -> <=, length(T))  # length(T) only works for StaticArrays, that's fine
default_signs(::Type{<:Real}) = <=

ecdf(data; signs=default_signs(eltype(data))) = ECDF(data, signs)

function (ecdf::ECDF)(x)
	sum(y -> is_all_leq(ecdf.signs, y, x), ecdf.data) / length(ecdf.data)
end

function (ecdf::ECDF)(x, ::typeof(count))
	sum(y -> is_all_leq(ecdf.signs, y, x), ecdf.data)
end

function (ecdf::ECDF)(x, stat::OnlineStat)
	fit!(copy(stat), filter(y -> is_all_leq(ecdf.signs, y, x), ecdf.data))
end

Broadcast.broadcasted(ecdf::ECDF, g::RectiGrid) = ecdf.(g, count) ./ length(ecdf.data)

function Broadcast.broadcasted(ecdf::ECDF, g::RectiGrid, ::typeof(count))
	axkeys = hasnames(g) ? named_axiskeys(g) : axiskeys(g)
	axspecs = map(axkeys, select_if_possible(ecdf.signs, axkeys)) do ax, sign
		@assert issorted(ax)
		ECDFAxisSpec(ax, sign)
	end
	counts = map(_ -> 0, g)
	for r in ecdf.data
		ix = findbin(axspecs, r) |> CartesianIndex
		checkbounds(Bool, counts, ix) || continue
		counts[ix] += 1
	end
	for (dim, ax) in enumerate(axspecs)
		aggregate_axis!(ax.sign, counts, dim)
	end
	return counts
end

function Broadcast.broadcasted(ecdf::ECDF, g::RectiGrid, stat::OnlineStat)
	axkeys = hasnames(g) ? named_axiskeys(g) : axiskeys(g)
	axspecs = map(axkeys, select_if_possible(ecdf.signs, axkeys)) do ax, sign
		@assert issorted(ax)
		ECDFAxisSpec(ax, sign)
	end
	result = map(_ -> copy(stat), g)
	for r in ecdf.data
		ix = findbin(axspecs, r) |> CartesianIndex
		checkbounds(Bool, result, ix) || continue
		fit!(result[ix], r)
	end
	for (dim, ax) in enumerate(axspecs)
		aggregate_axis!(ax.sign, result, dim)
	end
	return result
end

# like Base.AbstractVecOrTuple, but includes heterogeneous tuples
const AbstractVecOrTuple = Union{AbstractVector, Tuple}

@inline is_all_leq(sign, datap::Real, query::Real) = sign(datap, query)
@inline is_all_leq(sign, datap::Real, query::Base.Fix2) = query(datap)  # ignore `sign`
@inline is_all_leq(signs, datap, query) = all(map(is_all_leq, select_if_possible(signs, query), select_if_possible(datap, query), query))

# can specify a subset of fields for namedtuples...
select_if_possible(data::NamedTuple, query::NamedTuple{NSQ}) where {NSQ} = data[NSQ]
# but not for regular tuples
select_if_possible(data::AbstractVecOrTuple, query::AbstractVecOrTuple) = (@assert length(data) == length(query); data)

struct ECDFAxisSpec{TV, TO}
    binedges::TV
    sign::TO
end

@inline findbin(a::ECDFAxisSpec, x) = findbin(a.sign, a.binedges, x)

@inline findbin(as::Tuple, xs::Union{Tuple, AbstractVector}) = map((ax, x) -> findbin(ax, x), as, xs)
@inline findbin(as::NamedTuple{NS}, xs::NamedTuple) where {NS} = map((ax, x) -> findbin(ax, x), as, xs[NS]) |> values

@inline findbin(::typeof(<=), binedges::AbstractVector, x) = searchsortedfirst(binedges, x)
@inline findbin(::typeof(>=), binedges::AbstractVector, x) = searchsortedlast(binedges, x)

# aggregate_axis!(::Union{NoAggBelow, NoAggAbove}, A::AbstractArray, dim::Int) = A
aggregate_axis!(::typeof(<=), A::AbstractArray{<:Number}, dim::Int) = cumsum!(A, A, dims=dim)
aggregate_axis!(::typeof(<=), A::AbstractArray, dim::Int) = accumulate!(merge, A, A, dims=dim)
aggregate_axis!(::typeof(>=), A::AbstractArray, dim::Int) = (reverse!(A, dims=dim); cumsum!(A, A, dims=dim); reverse!(A, dims=dim))


end
