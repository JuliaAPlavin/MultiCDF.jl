module OnlineStatsBaseExt

using MultiCDF: ECDF, is_all_leq, select_if_possible, ECDFAxisSpec, findbin, aggregate_axis!
using MultiCDF.RectiGrids
using MultiCDF.AxisKeys
using MultiCDF.AxisKeys: hasnames
using OnlineStatsBase: OnlineStat, fit!, merge, value

function (ecdf::ECDF)(x, stat::OnlineStat)
	fit!(copy(stat), filter(y -> is_all_leq(ecdf.signs, y, x), ecdf.data))
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

end
