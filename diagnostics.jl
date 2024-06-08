using Oceananigans.AbstractOperations: @at, ∂x, ∂y, ∂z
using Oceananigans.Units
using Oceananigans.Grids: Center, Face
using Oceananigans.TurbulenceClosures: viscosity, diffusivity
using Oceananigans.Fields: @compute

using Oceanostics.FlowDiagnostics: strain_rate_tensor_modulus_ccc
using Oceanostics: KineticEnergyTendency, KineticEnergyDissipationRate, KineticEnergyStressTerm,
                   ErtelPotentialVorticity, DirectionalErtelPotentialVorticity, RossbyNumber, RichardsonNumber,
                   TracerVarianceDissipationRate, KineticEnergyForcingTerm, StrainRateTensorModulus, TurbulentKineticEnergy

#+++ Methods/functions definitions
keep(nt::NamedTuple{names}, keys) where names = NamedTuple{filter(x -> x ∈ keys, names)}(nt)
keep(d ::Dict, keys) = get.(Ref(d), keys, missing)
include("$(@__DIR__)/grid_metrics.jl")
include("$(@__DIR__)/budget.jl")

import Oceananigans.Fields: condition_operand
using Oceananigans.AbstractOperations: ConditionalOperation
using Oceananigans.Fields: AbstractField
using Oceananigans.Architectures: architecture, arch_array
using Oceananigans.ImmersedBoundaries: NotImmersed
@inline function condition_operand(func::Function, operand::AbstractField{<:Any, <:Any, <:Any, <:ImmersedBoundaryGrid}, condition::AbstractArray, mask)
    condition = NotImmersed(arch_array(architecture(operand.grid), condition))
    return ConditionalOperation(operand; func, condition, mask) 
end
#---

CellCenter = (Center, Center, Center)

#++++ Kernel Function Operations
using Oceananigans.Operators

#+++ Write to NCDataset
import NCDatasets as NCD
function write_to_ds(dsname, varname, data; coords=("xC", "yC", "zC"), dtype=Float64)
    ds = NCD.NCDataset(dsname, "a")
    if varname ∉ ds
        newvar = NCD.defVar(ds, varname, dtype, coords)
        newvar[:,:,:] = Array(data)
    end
    NCD.close(ds)
end
#---

#+++ Vector projection
@inline function vector_projection_aaa(i, j, k, grid, ϕˣ, ϕᶻ, params)
    return @inbounds ϕˣ[i,j,k]*params.xdirection + ϕᶻ[i,j,k]*params.zdirection
end
#---

#+++ Buoyancy Reynolds number
@inline buoyancy_reynolds_number_ccc(i, j, k, grid, u, v, w, N²) = 2*strain_rate_tensor_modulus_ccc(i, j, k, grid, u, v, w)^2 / N²
#---
#---

#+++ Define Fields
using Oceananigans.AbstractOperations: AbstractOperation
import Oceananigans.Fields: Field

ccc_scratch = Field{Center, Center, Center}(model.grid)
ScratchedField(op::AbstractOperation{Center, Center, Center}) = Field(op, data=ccc_scratch.data)

ScratchedField(f::Field) = f
ScratchedField(d::Dict) = Dict( k => ScratchedField(v) for (k, v) in d )
#---

#++++ Unpack model variables
u, v, w = model.velocities
b = model.tracers.b

outputs_vels = Dict{Any, Any}(:u => (@at CellCenter u),
                              :v => (@at CellCenter v),
                              :w => (@at CellCenter w),)
outputs_state_vars = merge(outputs_vels, Dict{Any, Any}(:b => b))
#---

#++++ CREATE SNAPSHOT OUTPUTS
#+++ Start calculation of snapshot variables
@info "Calculating misc diagnostics"

dbdx = @at CellCenter ∂x(b)
dbdy = @at CellCenter ∂y(b)
dbdz = @at CellCenter ∂z(b)

PV_x = @at CellCenter DirectionalErtelPotentialVorticity(model, (1, 0, 0))
PV_y = @at CellCenter DirectionalErtelPotentialVorticity(model, (0, 1, 0))
PV_z = @at CellCenter DirectionalErtelPotentialVorticity(model, (0, 0, 1))

ω_y = @at CellCenter (∂z(u) - ∂x(w))

εₖ = @at CellCenter KineticEnergyDissipationRate(model)
εₚ = @at CellCenter TracerVarianceDissipationRate(model, :b)/(2params.N2_inf)
εₛ = @at CellCenter KineticEnergyForcingTerm(model)

κₑ = diffusivity(model.closure, model.diffusivity_fields, Val(:b))
κₑ = κₑ isa Tuple ? sum(κₑ) : κₑ

Re_b = KernelFunctionOperation{Center, Center, Center}(buoyancy_reynolds_number_ccc, model.grid, u, v, w, params.N²∞)

Ri = @at CellCenter RichardsonNumber(model, u, v, w, b)
Ro = @at CellCenter RossbyNumber(model)
PV = @at CellCenter ErtelPotentialVorticity(model, u, v, w, b, model.coriolis)

outputs_dissip = Dict(pairs((;εₖ, εₚ, κₑ)))

outputs_misc = Dict(pairs((; dbdx, dbdy, dbdz, ω_y,
                             εₛ, Re_b,
                             Ri, Ro,
                             PV, PV_x, PV_y, PV_z,)))
#---

#+++ Define covariances
@info "Calculating covariances"
outputs_covs = Dict{Symbol, Any}(:uu => (@at CellCenter u*u),
                                 :vv => (@at CellCenter v*v),
                                 :ww => (@at CellCenter w*w),
                                 :uv => (@at CellCenter u*v),
                                 :uw => (@at CellCenter u*w),
                                 :vw => (@at CellCenter v*w),)
#---

#+++ Define velocity gradient tensor
@info "Calculating velocity gradient tensor"
outputs_grads = Dict{Symbol, Any}(:∂u∂x => (@at CellCenter ∂x(u)),
                                  :∂v∂x => (@at CellCenter ∂x(v)),
                                  :∂w∂x => (@at CellCenter ∂x(w)),
                                  :∂u∂y => (@at CellCenter ∂y(u)),
                                  :∂v∂y => (@at CellCenter ∂y(v)),
                                  :∂w∂y => (@at CellCenter ∂y(w)),
                                  :∂u∂z => (@at CellCenter ∂z(u)),
                                  :∂v∂z => (@at CellCenter ∂z(v)),
                                  :∂w∂z => (@at CellCenter ∂z(w)),)

@info "Calculating geostrophic velocity gradient tensor"
Uᵍ = @at CellCenter -∂y(model.pressures.pHY′) / params.f₀
Vᵍ = @at CellCenter +∂x(model.pressures.pHY′) / params.f₀ + params.V∞

outputs_geo_grads = Dict{Symbol, Any}(:∂Uᵍ∂x => (@at CellCenter ∂x(Uᵍ)),
                                      :∂Vᵍ∂x => (@at CellCenter ∂x(Vᵍ)),
                                      :∂Uᵍ∂y => (@at CellCenter ∂y(Uᵍ)),
                                      :∂Vᵍ∂y => (@at CellCenter ∂y(Vᵍ)),
                                      :∂Uᵍ∂z => (@at CellCenter ∂z(Uᵍ)),
                                      :∂Vᵍ∂z => (@at CellCenter ∂z(Vᵍ)),)
#---

#+++ Define energy budget terms
@info "Calculating energy budget terms"
outputs_budget = Dict{Symbol, Any}(:uᵢGᵢ     => KineticEnergyTendency(model),
                                   :uᵢ∂ⱼuⱼuᵢ => AdvectionTerm(model),
                                   :uᵢ∂ᵢp    => PressureTransportTerm(model, pressure = sum(model.pressures)),
                                   :uᵢbᵢ     => BuoyancyConversionTerm(model),
                                   :uᵢ∂ⱼτᵢⱼ  => KineticEnergyStressTerm(model),
                                   :uᵢ∂ⱼτᵇᵢⱼ => KineticEnergyImmersedBoundaryTerm(model),
                                   :εₛ       => εₛ,
                                   :Ek       => TurbulentKineticEnergy(model, u, v, w),)
#---

#+++ Assemble the "full" outputs tuple
@info "Assemble diagnostics quantities"
outputs_full = merge(outputs_state_vars, outputs_dissip, outputs_misc, outputs_grads, outputs_budget, outputs_geo_grads)
#---
#---

#+++ Construct outputs into simulation
function construct_outputs(simulation; 
                           simname = "TEST",
                           rundir = @__DIR__,
                           params = params,
                           overwrite_existing = overwrite_existing,
                           interval_2d = 0.2*params.T_advective,
                           interval_3d = params.T_advective,
                           interval_time_avg = 10*params.T_advective,
                           write_xyz = false,
                           write_xiz = true,
                           write_xyi = false,
                           write_iyz = false,
                           write_ttt = false,
                           write_tti = false,
                           write_aaa = false,
                           debug = false,
                           )
    #+++ get outputs
    model = simulation.model
    #---

    #+++ Get prefixes for conditional averages/integrals
    prefixes = (:∫∫⁰, :∫∫⁵, :∫∫¹⁰, :∫∫²⁰)
    buffers = [0, 5,]
    conditionally_integrated_var_symbols = (:εₖ, :εₚ, :uᵢbᵢ)
    #---

    #+++ Preamble and common keyword arguments
    k_half = @allowscalar Int(ceil(params.H / minimum_zspacing(grid))) # Approximately half the headland height
    kwargs = (overwrite_existing = overwrite_existing,
              deflatelevel = 5,
              global_attributes = merge(params, (; buffers)))
    #---

    #+++ xyz SNAPSHOTS
    if write_xyz
        @info "Setting up xyz writer"
        simulation.output_writers[:nc_xyz] = ow = NetCDFOutputWriter(model, ScratchedField(outputs_full);
                                                                     filename = "$rundir/data/xyz.$(simname).nc",
                                                                     schedule = TimeInterval(interval_3d),
                                                                     array_type = Array{Float64},
                                                                     verbose = debug,
                                                                     kwargs...
                                                                     )
        @info "Starting to write grid metrics and deltas to xyz"
        laptimer()
        add_grid_metrics_to!(ow)
        write_to_ds(ow.filepath, "Δx_from_headland", interior(compute!(Field(Δx_from_headland))), coords = ("xC", "yC", "zC"))
        write_to_ds(ow.filepath, "Δz_from_headland", interior(compute!(Field(Δz_from_headland))), coords = ("xC", "yC", "zC"))
        write_to_ds(ow.filepath, "altitude", interior(compute!(Field(altitude))), coords = ("xC", "yC", "zC"))
        write_to_ds(ow.filepath, "ΔxΔz", interior(compute!(Field(ΔxΔz))), coords = ("xC", "yC", "zC"))
        @info "Finished writing grid metrics and deltas to xyz"
        laptimer()
    end
    #---

    #+++ xyi SNAPSHOTS
    if write_xyi
        @info "Setting up xyi writer"
        indices = (:, :, k_half)
        outputs_xyi = outputs_full

        if write_aaa
            outputs_budget_integrated = Dict( Symbol(:∫∫∫, k, :dxdydz) => Integral(ScratchedField(v))  for (k, v) in outputs_budget )
            outputs_xyi = merge(outputs_xyi, outputs_budget_integrated)
        end

        simulation.output_writers[:nc_xyi] = ow = NetCDFOutputWriter(model, outputs_xyi;
                                                                     filename = "$rundir/data/xyi.$(simname).nc",
                                                                     schedule = TimeInterval(interval_2d),
                                                                     array_type = Array{Float64},
                                                                     indices = indices,
                                                                     verbose = debug,
                                                                     kwargs...
                                                                     )
        add_grid_metrics_to!(ow, user_indices=indices)
    end
    #---

    #+++ xiz (low def) SNAPSHOTS
    if write_xiz
        @info "Setting up xiz writer"
        indices = (:, grid.Ny÷2, :)
        simulation.output_writers[:nc_xiz] = ow = NetCDFOutputWriter(model, outputs_full;
                                                                     filename = "$rundir/data/xiz.$(simname).nc",
                                                                     schedule = TimeInterval(interval_2d),
                                                                     array_type = Array{Float32},
                                                                     indices = indices,
                                                                     verbose = debug,
                                                                     kwargs...
                                                                     )

        add_grid_metrics_to!(ow; user_indices=indices)
    end
    #---

    #+++ iyz (low def) SNAPSHOTS
    if write_iyz
        @info "Setting up iyz writer"
        indices = (ceil(Int, 4*grid.Nx/5), :, :)
        simulation.output_writers[:nc_iyz] = ow = NetCDFOutputWriter(model, outputs_full;
                                                                     filename = "$rundir/data/iyz.$(simname).nc",
                                                                     schedule = TimeInterval(interval_2d),
                                                                     array_type = Array{Float32},
                                                                     indices = indices,
                                                                     verbose = debug,
                                                                     kwargs...
                                                                     )
        add_grid_metrics_to!(ow, user_indices=indices)
    end
    #---

    #+++ ttt (Time averages)
    if write_ttt
        @info "Setting up ttt writer"
        outputs_ttt = merge(outputs_state_vars, outputs_covs, outputs_grads, outputs_dissip, outputs_budget)
        vp = @at CellCenter (v * sum(model.pressures))
        outputs_ttt = merge(outputs_ttt, Dict(:vp => vp, :p => sum(model.pressures)))
        indices = (:, :, :)
        simulation.output_writers[:nc_ttt] = ow = NetCDFOutputWriter(model, outputs_ttt;
                                                                     filename = "$rundir/data/ttt.$(simname).nc",
                                                                     schedule = AveragedTimeInterval(interval_time_avg, stride=10),
                                                                     array_type = Array{Float64},
                                                                     with_halos = false,
                                                                     indices = indices,
                                                                     verbose = true,
                                                                     kwargs...
                                                                     )
        add_grid_metrics_to!(ow, user_indices=indices)
        write_to_ds(ow.filepath, "altitude", interior(compute!(Field(altitude, indices=indices))), coords = ("xC", "yC", "zC"))
    end
    #---

    #+++ tti (Time averages)
    if write_tti
        @info "Setting up tti writer"
        outputs_tti = outputs_full
        indices = (:, :, k_half)
        simulation.output_writers[:nc_tti] = ow = NetCDFOutputWriter(model, outputs_tti;
                                                                     filename = "$rundir/data/tti.$(simname).nc",
                                                                     schedule = AveragedTimeInterval(interval_time_avg, stride=10),
                                                                     array_type = Array{Float64},
                                                                     with_halos = false,
                                                                     indices = indices,
                                                                     verbose = debug,
                                                                     kwargs...
                                                                     )
        add_grid_metrics_to!(ow, user_indices=indices)
    end
    #---

    #+++ Checkpointer
    @info "Setting up chk writer"
    simulation.output_writers[:chk_writer] = checkpointer = 
                                             Checkpointer(model;
                                             dir="$rundir/data/",
                                             prefix = "chk.$(simname)",
                                             schedule = TimeInterval(interval_time_avg),
                                             overwrite_existing = true,
                                             cleanup = true,
                                             verbose = debug,
                                             )
    #---

    return checkpointer
end
#---

