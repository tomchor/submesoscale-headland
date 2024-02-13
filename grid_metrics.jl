using Oceananigans.Grids: xspacing, yspacing, zspacing, xnode, ynode, znode, boundary_node, inactive_node, destantiate

flip_loc(::Center) = Face()
flip_loc(::Face) = Center()

boundary_xspacing(i,j,k,grid, ℓx, ℓy, ℓz) = ifelse(i==1, xnode(i,j,k, grid, flip_loc(ℓx), ℓy, ℓz) - xnode(i,j,k, grid, ℓx, ℓy, ℓz),
                                                         xnode(i,j,k, grid, ℓx, ℓy, ℓz)           - xnode(i-1,j,k, grid, flip_loc(ℓx), ℓy, ℓz))

boundary_yspacing(i,j,k,grid, ℓx, ℓy, ℓz) = ifelse(j==1, ynode(i,j,k, grid, ℓx, flip_loc(ℓy), ℓz) - ynode(i,j,k, grid, ℓx, ℓy, ℓz),
                                                         ynode(i,j,k, grid, ℓx, ℓy, ℓz)           - ynode(i,j-1,k, grid, ℓx, flip_loc(ℓy), ℓz))

boundary_zspacing(i,j,k,grid, ℓx, ℓy, ℓz) = ifelse(k==1, znode(i,j,k, grid, ℓx, ℓy, flip_loc(ℓz)) - znode(i,j,k, grid, ℓx, ℓy, ℓz),
                                                         znode(i,j,k, grid, ℓx, ℓy, ℓz)           - znode(i,j,k-1, grid, ℓx, ℓy, flip_loc(ℓz)))
for dir in (:x, :y, :z)
    active_spacing = Symbol(:active_, dir, :spacing)
    spacing = Symbol(dir, :spacing)
    boundary_spacing = Symbol(:boundary_, dir, :spacing)
    @eval begin
        function $active_spacing(i, j, k, grid, ℓx, ℓy, ℓz)
            Δξ = ifelse(inactive_node(i, j, k, grid, ℓx, ℓy, ℓz),
                        0, 
                        ifelse(boundary_node(i, j, k, grid, ℓx, ℓy, ℓz),
                               $boundary_spacing(i, j, k, grid, ℓx, ℓy, ℓz),
                               $spacing(i, j, k, grid, ℓx, ℓy, ℓz)))
            return Δξ
        end
    end
end

 

#+++ Output definitions
import Oceananigans.OutputWriters: save_output!, define_output_variable!
using Oceananigans.OutputWriters: fetch_and_convert_output, drop_output_dims,
                                  netcdf_spatial_dimensions, output_indices
using Oceananigans.Fields: AbstractField
using NCDatasets: defVar

define_timeconstant_variable!(dataset, output::AbstractField, name, array_type, compression, output_attributes, dimensions) =
    defVar(dataset, name, eltype(array_type), netcdf_spatial_dimensions(output),
           compression=compression, attrib=output_attributes)

function save_output!(ds, output, model, ow, name)

    data = fetch_and_convert_output(output, model, ow)
    data = drop_output_dims(output, data)
    colons = Tuple(Colon() for _ in 1:ndims(data))
    ds[name][colons...] = data
    return nothing
end
#---

function grid_metric_locations(outputs)
    location_list = []
    for output in values(outputs)
        loc = location(output)
        if (loc ∉ location_list) && (Nothing ∉ loc)
            push!(location_list, location(output))
        end
    end
    return location_list
end

loc_superscript(::Type{Center}) = "ᶜ"
loc_superscript(::Type{Face}) = "ᶠ"
loc_superscript(::Type{Nothing}) = "ᵃ"


function grid_spacings(grid::RectilinearGrid, ow)
    loc_list = unique(map(location, values(ow.outputs)))
    spacing_operations = Dict()

    for loc in loc_list
        LX, LY, LZ = loc

        Δx_name = "Δx" * loc_superscript(LX) * "ᵃᵃ"
        Δy_name = "Δyᵃ" * loc_superscript(LY) * "ᵃ"
        Δz_name = "Δzᵃᵃ" * loc_superscript(LZ)

        # Let's replace Nothing for Center for now since `active_xyzspacing` doesn't accept Nothig as location
        LX = LX == Nothing ? Center : LX
        LY = LY == Nothing ? Center : LY
        LZ = LZ == Nothing ? Center : LZ

        push!(spacing_operations, 
              Δx_name => Average(KernelFunctionOperation{LX, LY, LZ}(active_xspacing, grid, LX(), LY(), LZ()), dims=(2,3)),
              Δy_name => Average(KernelFunctionOperation{LX, LY, LZ}(active_yspacing, grid, LX(), LY(), LZ()), dims=(1,3)),
              Δz_name => Average(KernelFunctionOperation{LX, LY, LZ}(active_zspacing, grid, LX(), LY(), LZ()), dims=(1,2)))
    end
    return Dict(name => Field(op) for (name, op) in spacing_operations)
end


function grid_spacings(grid::ImmersedBoundaryGrid, ow)
    @info "adding grid spacings"
    loc_list = unique(map(location, values(ow.outputs)))
    spacing_operations = Dict()

    for loc in loc_list
        LX, LY, LZ = loc

        loc_3d = loc_superscript(LX) * loc_superscript(LY) * loc_superscript(LZ)
        Δx_name = "Δx" * loc_3d
        Δy_name = "Δy" * loc_3d
        Δz_name = "Δz" * loc_3d

        # Let's replace Nothing for Center for now since `active_xyzspacing` doesn't accept Nothing as location
        LX = LX == Nothing ? Center : LX
        LY = LY == Nothing ? Center : LY
        LZ = LZ == Nothing ? Center : LZ

        push!(spacing_operations, 
              Δx_name => KernelFunctionOperation{LX, LY, LZ}(active_xspacing, grid, LX(), LY(), LZ()),
              Δy_name => KernelFunctionOperation{LX, LY, LZ}(active_yspacing, grid, LX(), LY(), LZ()),
              Δz_name => KernelFunctionOperation{LX, LY, LZ}(active_zspacing, grid, LX(), LY(), LZ()))
    end
    return Dict(name => Field(op) for (name, op) in spacing_operations)
end


function write_grid_metrics!(ow, metrics; user_indices = (:, :, :), with_halos=false)
    ds = open(ow)
    for (metric_name, metric_operation) in metrics
        indices = output_indices(metric_operation, metric_operation.grid, user_indices, with_halos)
        sliced_metric = Field(metric_operation, indices=indices)

        if metric_name ∉ ds
            define_timeconstant_variable!(ds, sliced_metric, metric_name, ow.array_type, 0, Dict(), ("xC", "yC", "zC"))
            save_output!(ds, sliced_metric, model, ow, metric_name)
        end
    end
    close(ds)
end

function add_grid_metrics_to!(ow; kwargs...)
    Δξ_list = grid_spacings(grid, ow)
    write_grid_metrics!(ow, Δξ_list; kwargs...)
end

