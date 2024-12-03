using Oceananigans.Fields: ZeroField
using Oceananigans.Operators
using Oceananigans.Advection: div_𝐯u, div_𝐯v, div_𝐯w
using Oceananigans.TurbulenceClosures: immersed_∂ⱼ_τ₁ⱼ, immersed_∂ⱼ_τ₂ⱼ, immersed_∂ⱼ_τ₃ⱼ
@inline ψf(i, j, k, grid, ψ, f, args...) = @inbounds ψ[i, j, k] * f(i, j, k, grid, args...)

#+++ Advection term
@inline function uᵢ∂ⱼuⱼuᵢᶜᶜᶜ(i, j, k, grid, velocities, advection)
    u∂ⱼuⱼu = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, div_𝐯u, advection, velocities, velocities.u)
    v∂ⱼuⱼv = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, div_𝐯v, advection, velocities, velocities.v)
    w∂ⱼuⱼw = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, div_𝐯w, advection, velocities, velocities.w)
    return u∂ⱼuⱼu + v∂ⱼuⱼv + w∂ⱼuⱼw
end

function AdvectionTerm(model)
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ⱼuⱼuᵢᶜᶜᶜ, model.grid, model.velocities, model.advection)
end
#---

#+++ Immersed boundary stress term
@inline function immersed_uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ(i, j, k, grid,
                                            velocities,
                                            immersed_bcs,
                                            closure,
                                            diffusivity_fields,
                                            clock,
                                            model_fields)

    u∂ⱼ_τ₁ⱼ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, model_fields.u, immersed_∂ⱼ_τ₁ⱼ, velocities, immersed_bcs.u, closure, diffusivity_fields, clock, model_fields)
    v∂ⱼ_τ₂ⱼ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, model_fields.v, immersed_∂ⱼ_τ₂ⱼ, velocities, immersed_bcs.v, closure, diffusivity_fields, clock, model_fields)
    w∂ⱼ_τ₃ⱼ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, model_fields.w, immersed_∂ⱼ_τ₃ⱼ, velocities, immersed_bcs.w, closure, diffusivity_fields, clock, model_fields)

    return u∂ⱼ_τ₁ⱼ+ v∂ⱼ_τ₂ⱼ + w∂ⱼ_τ₃ⱼ
end

"""
Return a `KernelFunctionOperation` that computes the diffusive term of the KE prognostic equation:

```
    DIFF = uᵢ∂ⱼτᵢⱼ
```

where `uᵢ` are the velocity components and `τᵢⱼ` is the diffusive flux of `i` momentum in the 
`j`-th direction.
"""
function KineticEnergyImmersedBoundaryTerm(model; location = (Center, Center, Center))
    model_fields = fields(model)

    if model isa HydrostaticFreeSurfaceModel
        model_fields = (; model_fields..., w=ZeroField())
    end

    dependencies = (model.velocities, (u = model.velocities.u.boundary_conditions.immersed,
                                       v = model.velocities.v.boundary_conditions.immersed,
                                       w = model.velocities.w.boundary_conditions.immersed),
                    model.closure, 
                    model.diffusivity_fields, 
                    model.clock,
                    model_fields)
    return KernelFunctionOperation{Center, Center, Center}(immersed_uᵢ∂ⱼ_τᵢⱼᶜᶜᶜ, model.grid, dependencies...)
end
#---

#+++ Pressure transport term
@inline function uᵢ∂ᵢpᶜᶜᶜ(i, j, k, grid, velocities, pressure)
    u∂x_p = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, ∂xᶠᶜᶜ, pressure)
    v∂y_p = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, ∂yᶜᶠᶜ, pressure)
    w∂z_p = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, ∂zᶜᶜᶠ, pressure)
    return u∂x_p + v∂y_p + w∂z_p
end

function PressureTransportTerm(model; pressure = model.pressures.pNHS, location = (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ᵢpᶜᶜᶜ, model.grid, model.velocities, pressure)
end

@inline function uᵢ∂ᵢpHYᶜᶜᶜ(i, j, k, grid, velocities, pressure)
    u∂x_p = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, ∂xᶠᶜᶜ, pressure)
    v∂y_p = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, ∂yᶜᶠᶜ, pressure)
    return u∂x_p + v∂y_p
end

function HydrostaticPressureTransportTerm(model; pressure = model.pressures.pHY′, location = (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(uᵢ∂ᵢpHYᶜᶜᶜ, model.grid, model.velocities, pressure)
end
#---

#+++ Buoyancy conversion term
using Oceananigans.BuoyancyModels: x_dot_g_bᶠᶜᶜ, y_dot_g_bᶜᶠᶜ, z_dot_g_bᶜᶜᶠ
@inline function uᵢbᵢᶜᶜᶜ(i, j, k, grid, velocities, buoyancy_model, tracers)
    ubˣ = ℑxᶜᵃᵃ(i, j, k, grid, ψf, velocities.u, x_dot_g_bᶠᶜᶜ, buoyancy_model, tracers)
    vbʸ = ℑyᵃᶜᵃ(i, j, k, grid, ψf, velocities.v, y_dot_g_bᶜᶠᶜ, buoyancy_model, tracers)
    wbᶻ = ℑzᵃᵃᶜ(i, j, k, grid, ψf, velocities.w, z_dot_g_bᶜᶜᶠ, buoyancy_model, tracers)
    return ubˣ + vbʸ + wbᶻ
end

function BuoyancyConversionTerm(model; location = (Center, Center, Center), include_vertical_component = true)
    if !include_vertical_component # If the pressure is separated into hydrostatic + nonhydrostatic, then the vertical component is already taken into account
        model.velocities[:w] = ZeroField()
    end
    return KernelFunctionOperation{Center, Center, Center}(uᵢbᵢᶜᶜᶜ, model.grid, model.velocities, model.buoyancy, model.tracers)
end
#---

#+++ Kinetic Energy tendency
@inline ψζ(i, j, k, grid, ψ, ζ) = @inbounds ψ[i, j, k] * ζ[i, j, k]
@inline function uᵢG⁻ᵢᶜᶜᶜ(i, j, k, grid, velocities, G⁻)
        uG⁻ᵘ = ℑxᶜᵃᵃ(i, j, k, grid, ψζ, velocities.u, G⁻.u)
        vG⁻ᵛ = ℑxᶜᵃᵃ(i, j, k, grid, ψζ, velocities.v, G⁻.v)
        wG⁻ʷ = ℑxᶜᵃᵃ(i, j, k, grid, ψζ, velocities.w, G⁻.w)
    return uG⁻ᵘ + vG⁻ᵛ + wG⁻ʷ
end
function KineticEnergyTendency_G⁻(model::NonhydrostaticModel; location = (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(uᵢG⁻ᵢᶜᶜᶜ, model.grid, model.velocities, model.timestepper.G⁻)
end

@inline function uᵢGⁿᵢᶜᶜᶜ(i, j, k, grid, velocities, Gⁿ)
        uGⁿᵘ = ℑxᶜᵃᵃ(i, j, k, grid, ψζ, velocities.u, Gⁿ.u)
        vGⁿᵛ = ℑxᶜᵃᵃ(i, j, k, grid, ψζ, velocities.v, Gⁿ.v)
        wGⁿʷ = ℑxᶜᵃᵃ(i, j, k, grid, ψζ, velocities.w, Gⁿ.w)
    return uGⁿᵘ + vGⁿᵛ + wGⁿʷ
end
function KineticEnergyTendency_Gⁿ(model::NonhydrostaticModel; location = (Center, Center, Center))
    return KernelFunctionOperation{Center, Center, Center}(uᵢGⁿᵢᶜᶜᶜ, model.grid, model.velocities, model.timestepper.Gⁿ)
end
#---
