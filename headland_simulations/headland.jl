using DrWatson
using ArgParse
using Oceananigans
using Oceananigans.Units
using PrettyPrinting

using CUDA: @allowscalar, has_cuda_gpu

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

#+++ Base grid
grid_base = RectilinearGrid(topology = (Bounded, Periodic, Bounded),
                            size = (16, 20, 4),
                            x = (0, 800),
                            y = (0, 1000),
                            z = (0, 80),
                            )
#---

#+++ Immersed boundary
@inline east_wall(x, y, z) = x > 400
grid = ImmersedBoundaryGrid(grid_base, GridFittedBoundary(east_wall))
#---

#+++ Buoyancy model and background
buoyancy = BuoyancyTracer()

V∞ = 0.01
N²∞ = 6e-6
b∞(x, y, z) = N²∞ * z
#---

#+++ Model and ICs
@info "Creating model"
model = NonhydrostaticModel(grid = grid, timestepper = :RungeKutta3,
                            buoyancy = buoyancy, tracers = :b,
                            )
@info "" model

set!(model, b=b∞, v=0.01)
#---

#+++ Create simulation
simulation = Simulation(model, Δt=0.2*minimum_zspacing(grid.underlying_grid)/V∞,
                        stop_time=1e5,
                        )

wizard = TimeStepWizard(max_change=1.05, min_change=0.2, cfl=0.5, min_Δt=1e-4, max_Δt=1/√N²∞)
simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(2))

@info "" simulation
#---

outputs = (; model.pressures.pNHS,)
simulation.output_writers[:nc_xyz] = NetCDFOutputWriter(model, outputs;
                                                        filename = string(@__DIR__) * "/test_pressure.nc",
                   schedule = TimeInterval(200),
                   overwrite_existing = true,)

run!(simulation)
