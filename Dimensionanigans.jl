using Oceananigans
using DimensionalData
using Rasters

import DimensionalData: DimArray
import Rasters: Raster

function DimArray(f::Field; kwargs...)
    X, Y, Z = nodes(f)
    return DimArray(interior(f), (; X, Y, Z), kwargs...)
end

function DimArray(model::Oceananigans.AbstractModel; kwargs...)
    return map(DimArray, fields(model))
end

function Raster(f::Field; kwargs...)
    X, Y, Z = nodes(f)
    return Raster(DimArray(f; kwargs...))
end

function Raster(model::Oceananigans.AbstractModel; kwargs...)
    return map(Raster, fields(model))
end

