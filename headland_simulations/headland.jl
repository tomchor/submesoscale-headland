using DrWatson
using ArgParse
using Oceananigans
using Oceananigans.Units
using PrettyPrinting

using CUDA: @allowscalar, has_cuda_gpu

#+++ Preamble
#+++ Parse inital arguments
"Returns a dictionary of command line arguments."
function parse_command_line_arguments()
    settings = ArgParseSettings()
    @add_arg_table! settings begin

        "--simname"
            help = "Setup and name of simulation in siminfo.jl"
            default = "NPN-TEST-f32"
            arg_type = String

    end
    return parse_args(settings)
end
args = parse_command_line_arguments()
simname = args["simname"]
rundir = "$(DrWatson.findproject())/headland_simulations"
#---

#+++ Figure out name, dimensions, modifier, etc
sep = "-"
global topology, configname, modifiers... = split(simname, sep)
global AMD = "AMD" in modifiers ? true : false
global DNS = "DNS" in modifiers ? true : false
global f2  = "f2"  in modifiers ? true : false
global f4  = "f4"  in modifiers ? true : false
global f8  = "f8"  in modifiers ? true : false
global f16 = "f16" in modifiers ? true : false
global f32 = "f32" in modifiers ? true : false
global f64 = "f64" in modifiers ? true : false
global south = "S" in modifiers ? true : false
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
#---

#+++ Get primary simulation parameters
include("$(@__DIR__)/siminfo.jl")
params = getproperty(Headland(), Symbol(configname))
#---

#+++ Get secondary parameters
params = expand_headland_parameters(params)
#---
#---

#+++ Base grid
topo = (Bounded, Periodic, Bounded)
grid_base = RectilinearGrid(arch, topology = topo,
                            size = (14, 43, 4),
                            x = (-400, 400),
                            y = (-1000, 1000),
                            z = (0, 84),
                            halo = (4,4,4),
                            )
#---

#+++ Immersed boundary
include("bathymetry.jl")
@inline headland(x, y, z) = x > headland_x_of_yz(y, z, params)

GFB = GridFittedBoundary(headland)
grid = ImmersedBoundaryGrid(grid_base, GFB)
#---

#+++ Buoyancy model and background
buoyancy = BuoyancyTracer()

b∞(x, y, z, t, p) = p.N²∞ * z
#---

#+++ Sponge layer definition
f_params = (; params.H, params.L, params.sponge_length_y, params.V∞, params.f₀, params.N²∞,)
#---

#+++ Model and ICs
@info "Creating model"
model = NonhydrostaticModel(grid = grid, timestepper = :RungeKutta3,
                            advection = WENO(grid=grid_base, order=5),
                            buoyancy = buoyancy,
                            tracers = :b,
                            )
@info "" model

set!(model, b=(x, y, z) -> b∞(x, y, z, 0, f_params), v=params.V_inf)
#---

#+++ Create simulation
params = (; params..., T_advective_max = params.T_advective_spinup + params.T_advective_statistics)
simulation = Simulation(model, Δt=0.2*minimum_zspacing(grid.underlying_grid)/params.V_inf,
                        stop_time=5*params.T_advective,
                        wall_time_limit=23.2hours,
                        )

wizard = TimeStepWizard(max_change=1.05, min_change=0.2, cfl=0.5, min_Δt=1e-4, max_Δt=1/√params.N²∞)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))

@info "" simulation
#---

δ = ∂x(u) + ∂y(v) + ∂z(w)
outputs = (; δ, model.pressures.pNHS, model.pressures.pHY′)
interval_3d = 0.05*params.T_advective
simulation.output_writers[:nc_xyz] = NetCDFOutputWriter(model, outputs;
                   filename = "$rundir/data/xyz.$(simname).nc",
                   schedule = TimeInterval(0.01*params.T_advective),
                   overwrite_existing = true,)

run!(simulation)
