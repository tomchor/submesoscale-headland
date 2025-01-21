if ("PBS_JOBID" in keys(ENV))  @info "Job ID" ENV["PBS_JOBID"] end # Print job ID if this is a PBS simulation
using DrWatson
using ArgParse
using Oceananigans
using Oceananigans.Units
using PrettyPrinting
using TickTock

using CUDA: @allowscalar, has_cuda_gpu

#+++ Preamble
#+++ Parse inital arguments
"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()
    @add_arg_table! settings begin

        "--simname"
            help = "Setup and name of simulation in siminfo.jl"
            default = "NPN-TEST-f64"
            arg_type = String

    end
    return parse_args(settings)
end
args = parse_command_line_arguments()
simname = args["simname"]
rundir = @__DIR__
#---

#+++ Figure out name, dimensions, modifier, etc
sep = "-"
global topology, configname, modifiers... = split(simname, sep)
global f2  = "f2"  in modifiers ? true : false
global f4  = "f4"  in modifiers ? true : false
global f8  = "f8"  in modifiers ? true : false
global f16 = "f16" in modifiers ? true : false
global f32 = "f32" in modifiers ? true : false
global f64 = "f64" in modifiers ? true : false
global AMD = "AMD" in modifiers ? true : false
global south = "S" in modifiers ? true : false
global V2  =  "V2" in modifiers ? true : false
#---

#+++ Modify factor accordingly
if f2
    factor = 2
elseif f4
    factor = 4
elseif f8
    factor = 8
elseif f16
    factor = 16
elseif f32
    factor = 32
elseif f64
    factor = 64
else
    factor = 1
end
#---

#+++ Figure out architecture
if has_cuda_gpu()
    arch = GPU()
else
    arch = CPU()
end
@info "Starting simulation $simname with a dividing factor of $factor and a $arch architecture\n"
#---

#+++ Get primary simulation parameters
include("$(@__DIR__)/../siminfo.jl")
params = getproperty(Headland(), Symbol(configname))

if V2
    params = (; params..., V∞ = 2*params.V∞)
end
#---

#+++ Get secondary parameters
params = expand_headland_parameters(params)

if south
    params = (; params..., f_0 = -params.f_0, f₀ = -params.f₀)
end

simname_full = simname
@info "Nondimensional parameter space" params.Ro_h params.Fr_h params.α params.Bu_h params.Γ 
@info "Dimensional parameters" params.L params.H params.N²∞ params.f₀ params.z₀
pprintln(params)
#---
#---

#+++ Base grid
#+++ Figure out topology and domain
if topology == "NPN"
    topo = (Bounded, Periodic, Bounded)
else
    throw(AssertionError("Topology must be NPN"))
end
#---

params = (; params..., factor)

if AMD # AMD takes up more memory...
    params = (; params..., N=params.N*0.85)
end

NxNyNz = get_sizes(params.N ÷ (factor^3),
                   Lx=params.Lx, Ly=params.Ly, Lz=params.Lz,
                   aspect_ratio_x=4.2, aspect_ratio_y=3.5)

params = (; params..., NxNyNz...)

grid_base = RectilinearGrid(arch, topology = topo,
                            size = (params.Nx, params.Ny, params.Nz),
                            x = (-params.Lx + params.headland_intrusion_size_max, +params.headland_intrusion_size_max),
                            y = (-params.y_offset, params.Ly-params.y_offset),
                            z = (0, params.Lz),
                            halo = (4,4,4),
                            )
@info grid_base
params = (; params..., Δz_min = minimum_zspacing(grid_base))
#---

#+++ Immersed boundary
include("../bathymetry.jl")
@inline headland(x, y, z) = x > headland_x_of_yz(y, z, params)

#+++ Bathymetry visualization
if false

    bathymetry2(x, y) = headland_x_of_yz(x, y, params.H) # For visualization purposes
    bathymetry3(y) = headland_x_of_yz(0, y, 0) # For visualization purposes

    xc = xnodes(grid_base, Center())
    yc = ynodes(grid_base, Center())
    zc = znodes(grid_base, Center())

    using GLMakie
    lines(yc, bathymetry3)
    pause
end

if false
    bathymetry2(x, y, z) = headland(x, y, z)

    xc = xnodes(grid_base, Center())
    yc = ynodes(grid_base, Center())
    zc = znodes(grid_base, Center())

    using GLMakie

    volume(xc, yc, zc, bathymetry2,
           isovalue = 1, isorange = 0.5,
           algorithm = :iso,
           axis=(type=Axis3, aspect=(params.Lx, params.Ly, 5params.Lz)))
    pause
end
#---

GFB = GridFittedBoundary(headland)
grid = ImmersedBoundaryGrid(grid_base, GFB)
@info grid
#---

#+++ Drag (Implemented as in https://doi.org/10.1029/2005WR004685)
z₀ = params.z_0 # roughness length
z₁ = minimum_zspacing(grid_base, Center(), Center(), Center())/2
@info "Using z₁ =" z₁

const κᵛᵏ = 0.4 # von Karman constant
params = (; params..., c_dz = (κᵛᵏ / log(z₁/z₀))^2) # quadratic drag coefficient
@info "Defining momentum BCs with Cᴰ (x, y, z) =" params.c_dz

@inline τᵘ_drag(x, y, z, t, u, v, w, p) = -p.Cᴰ * u * √(u^2 + v^2 + w^2)
@inline τᵛ_drag(x, y, z, t, u, v, w, p) = -p.Cᴰ * v * √(u^2 + v^2 + w^2)
@inline τʷ_drag(x, y, z, t, u, v, w, p) = -p.Cᴰ * w * √(u^2 + v^2 + w^2)

τᵘ = FluxBoundaryCondition(τᵘ_drag, field_dependencies = (:u, :v, :w), parameters=(; Cᴰ = params.c_dz,))
τᵛ = FluxBoundaryCondition(τᵛ_drag, field_dependencies = (:u, :v, :w), parameters=(; Cᴰ = params.c_dz,))
τʷ = FluxBoundaryCondition(τʷ_drag, field_dependencies = (:u, :v, :w), parameters=(; Cᴰ = params.c_dz,))

u_bcs = FieldBoundaryConditions(immersed=τᵘ)
v_bcs = FieldBoundaryConditions(immersed=τᵛ)
w_bcs = FieldBoundaryConditions(immersed=τʷ)
#---

#+++ Buoyancy model and background
b∞(x, y, z, t, p) = p.N²∞ * z

b_bcs = FieldBoundaryConditions()

bcs = (u=u_bcs,
       v=v_bcs,
       w=w_bcs,
       b=b_bcs,
       )
#---

#+++ Sponge layer definition
@info "Defining sponge layer"
params = (; params..., y_south = ynode(1, grid, Face()))
mask_y_params = (; params.y_south, params.sponge_length_y, σ = params.sponge_rate)

const y₀ = params.y_south
const y₁ = y₀ + params.sponge_length_y/2
const y₂ = y₀ + params.sponge_length_y
@inline south_mask_linear(x, y, z, p) = ifelse((y₀ <= y <= y₁),
                                               (y-y₀)/(y₁-y₀),
                                               ifelse((y₁ <= y <= y₂),
                                                      (y-y₂)/(y₁-y₂), 0.0
                                                      ))

@inline sponge_u(x, y, z, t, u, p) = -(south_mask_linear(x, y, z, p)) * p.σ * u
@inline sponge_v(x, y, z, t, v, p) = -(south_mask_linear(x, y, z, p)) * p.σ * (v - p.V∞)
@inline sponge_w(x, y, z, t, w, p) = -(south_mask_linear(x, y, z, p)) * p.σ * w
@inline sponge_b(x, y, z, t, b, p) = -(south_mask_linear(x, y, z, p)) * p.σ * (b - b∞(0, 0, z, 0, p))

@inline geostrophy(x, y, z, p) = -p.f₀ * p.V∞

forc_u(x, y, z, t, u, p) = sponge_u(x, y, z, t, u, p) + geostrophy(x, y, z, p)
forc_v(x, y, z, t, v, p) = sponge_v(x, y, z, t, v, p)
forc_w(x, y, z, t, w, p) = sponge_w(x, y, z, t, w, p)
forc_b(x, y, z, t, b, p) = sponge_b(x, y, z, t, b, p)


f_params = (; params.H, params.L, params.sponge_length_y,
            params.V∞, params.f₀, params.N²∞,)

Fᵤ = Forcing(forc_u, field_dependencies = :u, parameters = merge(mask_y_params, (; f₀ = params.f_0, V∞ = params.V_inf)))
Fᵥ = Forcing(forc_v, field_dependencies = :v, parameters = merge(mask_y_params, f_params))
Fw = Forcing(forc_w, field_dependencies = :w, parameters = mask_y_params)
Fb = Forcing(forc_b, field_dependencies = :b, parameters = merge(mask_y_params, f_params))
#---

#+++ Turbulence closure
if AMD
    closure = AnisotropicMinimumDissipation()
else
    closure = SmagorinskyLilly(C=0.13, Pr=1)
end
#---

#+++ Model and ICs
@info "Creating model"
model = NonhydrostaticModel(grid = grid, timestepper = :RungeKutta3,
                            advection = WENO(grid=grid_base, order=5),
                            buoyancy = BuoyancyTracer(),
                            coriolis = FPlane(params.f_0),
                            tracers = :b,
                            closure = closure,
                            boundary_conditions = bcs,
                            forcing = (u=Fᵤ, v=Fᵥ, w=Fw, b=Fb),
                           )
@info "" model
if has_cuda_gpu() run(`nvidia-smi -i $(ENV["CUDA_VISIBLE_DEVICES"])`) end

set!(model, b=(x, y, z) -> b∞(x, y, z, 0, f_params), v=params.V_inf)
#---

#+++ Create simulation
params = (; params..., T_advective_max = params.T_advective_spinup + params.T_advective_statistics)
simulation = Simulation(model, Δt=0.1*minimum_zspacing(grid.underlying_grid)/params.V_inf,
                        stop_time=params.T_advective_max * params.T_advective,
                        wall_time_limit=23.5hours,
                        )

using Oceanostics.ProgressMessengers
walltime_per_timestep = StepDuration(with_prefix=false) # This needs to instantiated here, and not in the function below
walltime = Walltime()
progress(simulation) = @info (PercentageProgress(with_prefix=false, with_units=false)
                              + "$(round(time(simulation)/params.T_advective; digits=2)) adv periods" + walltime
                              + TimeStep() + MaxVelocities() + "CFL = "*AdvectiveCFLNumber(with_prefix=false)
                              + "step dur = "*walltime_per_timestep)(simulation)
simulation.callbacks[:progress] = Callback(progress, IterationInterval(40))

wizard = TimeStepWizard(max_change=1.05, min_change=0.2, cfl=0.95, min_Δt=1e-4, max_Δt=1/√params.N²∞)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))

@info "" simulation
#---

#+++ Diagnostics
#+++ Check for checkpoints
if any(startswith("chk.$(simname)_iteration"), readdir("$rundir/data"))
    @warn "Checkpoint for $simname found. Assuming this is a pick-up simulation! Setting overwrite_existing=false."
    overwrite_existing = false
else
    @warn "No checkpoint for $simname found. Setting overwrite_existing=true."
    overwrite_existing = true
end
#---

include("$rundir/../diagnostics.jl")
tick()
checkpointer = construct_outputs(simulation,
                                 simname = simname,
                                 rundir = rundir,
                                 params = params,
                                 overwrite_existing = overwrite_existing,
                                 interval_2d = 0.2*params.T_advective,
                                 interval_3d = 0.2*params.T_advective,
                                 interval_time_avg = 20*params.T_advective,
                                 write_xyz = true,
                                 write_xiz = false,
                                 write_xyi = false,
                                 write_iyz = false,
                                 write_ttt = false,
                                 write_tti = false,
                                 debug = false,
                                 )
tock()
#---

#+++ Run simulations and plot video afterwards
if has_cuda_gpu() run(`nvidia-smi -i $(ENV["CUDA_VISIBLE_DEVICES"])`) end
@info "Starting simulation"
run!(simulation, pickup=true)
#---

include(string(@__DIR__) * "/hplot_bathymetry.jl")
