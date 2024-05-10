using Oceananigans

grid_base = RectilinearGrid(topology = (Bounded, Periodic, Bounded),
                            size = (16, 20, 4), extent = (800, 1000, 100),)

@inline east_wall(x, y, z) = x > 400
grid = ImmersedBoundaryGrid(grid_base, GridFittedBoundary(east_wall))

model = NonhydrostaticModel(grid = grid, timestepper = :RungeKutta3,
                            buoyancy = BuoyancyTracer(), tracers = :b,
                            )

N² = 6e-6
b∞(x, y, z) = N² * z
set!(model, b=b∞)

simulation = Simulation(model, Δt=25, stop_time=1e4,)
Δt(model) = model.clock.last_Δt
simulation.output_writers[:snaps] = NetCDFOutputWriter(model, (; model.pressures.pNHS, dpdx=∂x(model.pressures.pNHS), Δt=Δt),
                                                       filename = "test_pressure.nc",
                                                       schedule = TimeInterval(100),
                                                       dimensions = (; Δt = ()),
                                                       overwrite_existing = true,)
run!(simulation)
