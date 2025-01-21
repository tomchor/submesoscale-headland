using Oceananigans.Fields: @compute
import Oceananigans.Grids: xnode, ynode, znode
xnode(i, grid, ℓx) = xnode(i, 1, 1, grid, ℓx, Center(), Center())
ynode(j, grid, ℓy) = ynode(1, j, 1, grid, Center(), ℓy, Center())
znode(k, grid, ℓz) = znode(1, 1, k, grid, Center(), Center(), ℓz)

#+++ Define headland as x(y, z)
@inline η(z, p) = 2*p.L + (0 - 2*p.L) * z / (2*p.H) # headland intrusion size
@inline headland_width(z, p) = p.β * η(z, p)
@inline headland_x_of_yz(y, z, p) = 2*p.L - η(z, p) * exp(-(2y / headland_width(z, p))^2)
#---

#+++ Now define it as z(x, y)
using LambertW: lambertw
ξ(x, p) = x - 2p.L
W(x, y, p) = lambertw((√8*y/(p.β*ξ(x, p)))^2)
f(x, y, p) = y == 0 ? 2ξ(x, p) / 4p.L : 2*√8*y / (4p.L * p.β * √W(x, y, p))
@inline headland_z_of_xy⁺(x, y, p) = 2*p.H * (1 + f(x, y, p))
@inline headland_z_of_xy⁻(x, y, p) = 2*p.H * (1 - f(x, y, p))
@inline headland_z_of_xy_unbounded(x, y, p) = min(headland_z_of_xy⁺(x, y, p), headland_z_of_xy⁻(x, y, p))
@inline headland_z_of_xy(x, y, p) = max(headland_z_of_xy_unbounded(x, y, p), 0)
#---

#+++ Now calculate approximate x, z distance
using Oceananigans.Grids: xnode, ynode, znode
params_geometry = (; params.H, params.Lx, params.β, params.L)
@inline x_distance_from_headland_boundary_ccc(i, j, k, grid, p) = headland_x_of_yz(ynode(j, grid, Center()), znode(k, grid, Center()), p) - xnode(i, grid, Center())
Δx_from_headland = KernelFunctionOperation{Center, Center, Center}(x_distance_from_headland_boundary_ccc, grid_base, params_geometry)

@inline z_distance_from_headland_boundary_ccc(i, j, k, grid, p) = znode(k, grid, Center()) -
                                                                  headland_z_of_xy_unbounded(xnode(i, grid, Center()), ynode(j, grid, Center()), p)
Δz_from_headland = KernelFunctionOperation{Center, Center, Center}(z_distance_from_headland_boundary_ccc, grid_base, params_geometry)

function altitude_ccc(i, j, k, grid, p)
    Δx = x_distance_from_headland_boundary_ccc(i, j, k, grid, p)
    Δz = z_distance_from_headland_boundary_ccc(i, j, k, grid, p)
    return sign(Δx) * √(1/(1/Δx^2 + 1/Δz^2))
end

@compute altitude = Field(KernelFunctionOperation{Center, Center, Center}(altitude_ccc, grid_base, params_geometry))
#---

xC = KernelFunctionOperation{Center, Center, Center}(xnode, grid_base, Center(), Center(), Center())
yC = KernelFunctionOperation{Center, Center, Center}(ynode, grid_base, Center(), Center(), Center())
zC = KernelFunctionOperation{Center, Center, Center}(znode, grid_base, Center(), Center(), Center())

using Oceananigans.Grids: xspacing, zspacing
ΔxΔz_kernel(i, j, k, grid, ℓx, ℓy, ℓz) = xspacing(i, j, k, grid, ℓx, ℓy, ℓz) * zspacing(i, j, k, grid, ℓx, ℓy, ℓz)
ΔxΔz = KernelFunctionOperation{Center, Center, Center}(ΔxΔz_kernel, grid_base, Center(), Center(), Center())

@inline headland_z_of_xy(i, j, k, grid, p) = headland_z_of_xy(xnode(i, grid, Center()), ynode(j, grid, Center()), p)
bottom_height = KernelFunctionOperation{Center, Center, Center}(headland_z_of_xy, grid_base, params)
