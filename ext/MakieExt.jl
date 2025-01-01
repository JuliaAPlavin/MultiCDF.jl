module MakieExt

using MultiCDF
using MultiCDF.DataPipes, MultiCDF.Accessors, MultiCDF.StructArrays
using MultiCDF: ECDF, ContiguousIntervals
using Makie
using Makie: mean
using Makie.IntervalSets
using MultiCDF.AxisKeys
using MultiCDF.RectiGrids

## 1d

Makie.used_attributes(T::Type{<:ScatterLines}, _, x::ECDF) = used_attributes(T, x)
Makie.used_attributes(T::Type{<:ScatterLines}, ::ECDF) = (:doaxis, Makie.attribute_names(T)...)

Makie.convert_arguments(ct::Type{<:ScatterLines}, E::ECDF; kwargs...) = convert_arguments(ct, nothing, E; kwargs...)

function Makie.convert_arguments(ct::Type{<:ScatterLines}, interval, E::ECDF; doaxis=false, kwargs...)
    pspec = convert_arguments(ct, interval, ECDF(map(only, E.data), only(E.signs)); kwargs...)
    if doaxis
        S = Makie.SpecApi
        S.GridLayout([S.Axis(plots=[pspec], xlabel=only(fieldnames(eltype(E.data))) |> string, ylabel="Fraction $(only(E.signs)) x")])
    else
        pspec
    end
end

function Makie.convert_arguments(ct::Type{<:ScatterLines}, interval, E::ECDF{<:Number}; doaxis=false, markersize=nothing, kwargs...)
    rev = Dict(
        (<) => false,
        (<=) => false,
        (>) => true,
        (>=) => true,
    )[E.signs]
    vs = sort(E.data; rev)
    interval = @something(interval, Interval(minmax(first(vs), last(vs))...))
    lrval = rev ? reverse(extrema(interval)) : extrema(interval)
    markersize = something(markersize, Makie.default_theme(nothing, ct).markersize[])
    pspec = Makie.to_plotspec(ct,
        (
            [first(lrval); repeat(vs; inner=2); last(lrval)],
            repeat((0:length(vs)) ./ length(vs), inner=2)
        );
        markersize=(@p [false; repeat([false, true], outer=length(vs)); false] map(_ ? markersize : zero(markersize))),
        kwargs...)
    if doaxis
        S = Makie.SpecApi
        S.GridLayout([S.Axis(plots=[pspec], ylabel="Fraction $(E.signs) x")])
    else
        pspec
    end
end


## 2d

function Makie.convert_arguments(ct::CellGrid, E::ECDF)
    axints = map(ecdf_axvals(E)) do vs
        ContiguousIntervals(vs)
    end
    EG = E.(grid(;map(a -> mean.(a), axints)...))
    EGi = @set named_axiskeys(EG) = axints
    (EGi,)
end

function Makie.convert_arguments(ct::CellGrid, xi::ClosedInterval, yi::ClosedInterval, E::ECDF)
    axints = map(ecdf_axvals(E)) do vs
        ContiguousIntervals(vs)
    end
    axints = @p let
        axints
        @modify(filter(∈(xi), _), __[1].edges)
        @modify(filter(∈(yi), _), __[2].edges)
        @insert first(__[1].edges) = minimum(xi)
        @insert last(__[1].edges) = maximum(xi)
        @insert first(__[2].edges) = minimum(yi)
        @insert last(__[2].edges) = maximum(yi)
    end
    EG = E.(grid(;map(a -> mean.(a), axints)...))
    EGi = @set named_axiskeys(EG) = axints
    (EGi,)
end

Makie.convert_arguments(ct::ImageLike, xi::ClosedInterval, yi::ClosedInterval, E::ECDF) =
    convert_arguments(ct, range(xi, length=500), range(yi, length=500), E)

function Makie.convert_arguments(ct::ImageLike, xi::AbstractRange, yi::AbstractRange, E::ECDF)
    axrngs = NamedTuple{keys(E.signs)}((xi, yi))
    (E.(grid(;axrngs...)),)
end

function ecdf_axvals(ecdf)
    sa = StructArray(ecdf.data)
    axvals = map(StructArrays.components(sa)) do comp
        vals = sort(comp) |> unique!
        uvals = unique(vals)
        if length(uvals) ≥ 2
            push!(vals, uvals[end] + (uvals[end] - uvals[end-1]))
            pushfirst!(vals, uvals[begin] + (uvals[begin] - uvals[begin+1]))
        end
        vals
    end
end

Makie.convert_arguments(ct::CellGrid, x::ContiguousIntervals, y::ContiguousIntervals, z::AbstractMatrix{<:Real}) =
    convert_arguments(ct, x.edges, y.edges, z)

end
