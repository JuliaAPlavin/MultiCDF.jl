module MultiCDF

export ecdf, ecdf_from_pdf, from_unit_cube

using RectiGrids
using OnlineStatsBase: OnlineStat, fit!, merge, value
using AxisKeys
using AxisKeys: hasnames


default_signs(::Type{<:NamedTuple{NS}}) where {NS} = NamedTuple{NS}(ntuple(_ -> <=, length(NS)))
default_signs(::Type{T}) where {T <: Tuple} = ntuple(_ -> <=, fieldcount(T))
default_signs(::Type{T}) where {T <: AbstractVector} = ntuple(_ -> <=, length(T))  # length(T) only works for StaticArrays, that's fine
default_signs(::Type{<:Real}) = <=


struct ECDF{T, TD <: AbstractVector{T}, TS}
	data::TD
	signs::TS
end

ecdf(data; signs=default_signs(eltype(data))) = ECDF(data, signs)

function (ecdf::ECDF)(x)
	count(y -> is_all_leq(ecdf.signs, y, x), ecdf.data) / length(ecdf.data)
end

function (ecdf::ECDF)(x, ::typeof(count))
	count(y -> is_all_leq(ecdf.signs, y, x), ecdf.data)
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

struct ECDF_PDF{TD, TS}
	data::TD
	signs::TS
end

Base.broadcastable(x::ECDF_PDF) = Ref(x)
ecdf_from_pdf(data::KeyedArray{T, 2}) where {T} = ECDF_PDF((cumsum(data; dims=2), cumsum(sum(data; dims=2)[:, 1])), nothing)

from_unit_cube(df::ECDF_PDF, x::Tuple) = from_unit_cube_(df.data, x)
from_unit_cube(df::ECDF_PDF, x::T) where {T} = T(from_unit_cube_(df.data, x))

@inline function from_unit_cube_(cum_margins::KeyedArray{T, 1}, x) where {T}
	ix, val = axkey_interp(cum_margins, x[1] * cum_margins[end])
	(val,)
end
@inline function from_unit_cube_((data, cum_margins)::Tuple, x)
	ix, val = axkey_interp(cum_margins, x[1] * cum_margins[end])
	(val, from_unit_cube_(view(data, ix, :), x[2:end])...)
end

@inline function axkey_interp(A::KeyedArray{T, 1}, x) where {T}
	ix = x == A[end] ?
		searchsortedfirst(A, x) :
		searchsortedlast(A, x) + 1
	curkey = axiskeys(A, 1)[ix]
	prevA = ix == 1 ? zero(eltype(A)) : A[ix - 1]
	prevkey = ix == 1 ? curkey - (axiskeys(A, 1)[ix + 1] - curkey) : axiskeys(A, 1)[ix - 1]
	t = (x - prevA) / (A[ix] - prevA)
	# t = isfinite(t) ? t : one(t)
	val = t * curkey + (1 - t) * prevkey
	return ix, val
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

@inline findbin(as::Tuple, xs::Tuple) = map((ax, x) -> findbin(ax, x), as, xs)
@inline findbin(as::Tuple, xs::AbstractVector) = findbin(as, Tuple(xs))
@inline findbin(as::NamedTuple{NS}, xs::NamedTuple) where {NS} = map((ax, x) -> findbin(ax, x), as, xs[NS]) |> values

@inline findbin(::typeof(<=), binedges::AbstractVector, x) = searchsortedfirst(binedges, x)
@inline findbin(::typeof(>=), binedges::AbstractVector, x) = searchsortedlast(binedges, x)
@inline findbin(::typeof(<), binedges::AbstractVector, x) = let
	ix = searchsortedfirst(binedges, x)
	ix <= lastindex(binedges) && binedges[ix] == x ?
		ix + 1 :
		ix
end
@inline findbin(::typeof(>), binedges::AbstractVector, x) = let
	ix = searchsortedlast(binedges, x)
	ix >= firstindex(binedges) && binedges[ix] == x ?
		ix - 1 :
		ix
end

const LT = Union{typeof(<=), typeof(<)}
const GT = Union{typeof(>=), typeof(>)}
aggregate_axis!(::LT, A::AbstractArray{<:Number}, dim::Int) = cumsum!(A, A, dims=dim)
aggregate_axis!(::LT, A::AbstractArray, dim::Int) = accumulate!(merge, A, A, dims=dim)
aggregate_axis!(::GT, A::AbstractArray, dim::Int) = (reverse!(A, dims=dim); cumsum!(A, A, dims=dim); reverse!(A, dims=dim))


end
