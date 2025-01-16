import Oceananigans.BoundaryConditions: _fill_north_halo!

@inline function _fill_east_halo!(j, k, grid, u, bc::PAOBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    i = grid.Nx + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i-1, j, k)
    Δx = Δxᶠᶜᶜ(i, j, k, grid)
    step_right_boundary!(bc, j, k, boundary_indices, boundary_adjacent_indices, grid, u, clock, model_fields, Δx)
    return nothing
end

@inline function _fill_north_halo!(i, k, grid, v, bc::PAOBC, ::Tuple{Face, Any, Any}, clock, model_fields)
    j = grid.Ny + 1
    boundary_indices = (i, j, k)
    boundary_adjacent_indices = (i, j-1, k)
    Δy = Δyᶜᶠᶜ(i, j, k, grid)
    step_right_boundary!(bc, i, k, boundary_indices, boundary_adjacent_indices, grid, v, clock, model_fields, Δy)
    return nothing
end

