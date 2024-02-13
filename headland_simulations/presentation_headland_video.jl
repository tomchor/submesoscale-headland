@info "Starting to plot video..."

#+++ Figure out if we have a screen or not (https://github.com/JuliaPlots/Plots.jl/issues/4368)
if ("GITHUB_ENV" ∈ keys(ENV)) || # Is it a github CI?
    ("PBS_JOBID" ∈ keys(ENV)  && !(ENV["PBS_ENVIRONMENT"] == "PBS_INTERACTIVE")) # Is it a non-interactive PBS job?
    @info "Headless server! (probably NCAR or github CI). Loading CairoMakie"
    using CairoMakie
    get!(ENV, "GKSwstype",  "nul")
elseif any(map(x -> startswith(x, "NCAR"), Tuple(keys(ENV)))) # Is this an NCAR server?
    @info "Detected that we're working on NCAR servers. Loading CairoMakie"
    using CairoMakie
else
    @info "Loading GLMakie"
    using GLMakie
end
#---

#+++ Read datasets
if @isdefined simulation
    fpath_xiz = simulation.output_writers[:nc_xiz].filepath
else
    simname_full = "NPN-R02F02"
    fpath_xiz = "data/xiz.$simname_full.nc"
end

using Rasters
using Rasters: name

function squeeze(ds::Union{Raster, RasterStack})
    flat_dimensions = NamedTuple((name(dim), 1) for dim in dims(ds) if length(dims(ds, dim)) ==  1)
    return getindex(ds; flat_dimensions...)
end

ds_xiz = RasterStack(fpath_xiz, lazy=true)

# Get other datasets
dslist = Vector{Any}([(ds_xiz, "xz")])
for (prefix, slice) in [ ("xyi", "xy"), ]
    fpath = replace(fpath_xiz, "xiz" => prefix)
    if isfile(fpath)
        pushfirst!(dslist, (RasterStack(fpath), slice))
    end
end

# Get the indices for the slices
slicelist = []
for (i, (ds, slice)) in enumerate(dslist)
    dim_index = if slice == "xy"
                    :zC
                elseif slice == "xz"
                    :yC
                elseif slice == "yz"
                    :xC
                elseif slice == "xyz"
                    nothing
                end
    dim_value = dims(ds, dim_index)[1]
    push!(slicelist, (slice, string(first(string(dim_index))), dim_value))
end
#---

#+++ Get parameters
if !((@isdefined params) && (@isdefined simulation))
    md = metadata(ds_xiz)
    params = (; (Symbol(k) => v for (k, v) in md)...)
end
#---

#+++ Auxiliary parameters
u_lims = (-params.V_inf, +params.V_inf) .* 1.2
w_lims = u_lims
PV_lims = params.N2_inf * params.f_0 * [-5, +5]
ε_max = maximum(ds_xiz.εₖ)
ε_lims = (ε_max/1e6, ε_max/1e2)
#---

#+++ Decide datasets, frames, etc.
times = dims(ds_xiz, :Ti)
n_times = length(times)
max_frames = 150
step = max(1, floor(Int, n_times / max_frames))

dslist = [ (squeeze(ds), slice) for (ds, slice) in dslist ]
#---

#+++ Plotting options
variables = (:PV, :Ro, :εₖ)

kwargs = Dict(:u => (colorrange = u_lims,
                     colormap = :balance),
              :v => (colorrange = u_lims,
                     colormap = :balance),
              :w => (colorrange = w_lims,
                     colormap = :balance),
              :PV => (colorrange = PV_lims,
                      colormap = :seismic),
              :εₖ => (colormap = :inferno,
                      colorscale = log10,
                      colorrange = ε_lims,),
              :Ro => (; colorrange = (-2, +2),
                      colormap = :balance),
              :Ri => (; colorrange = (-2, +2),
                      colormap = :balance),
              )

title_height = 8
panel_height = 140; panel_width = 300
cbar_height = 8
bottom_axis_height = 2panel_height/3
#---

#+++ Plotting preamble
using Oceananigans.Units, Printf
using Oceananigans: prettytime

fig = Figure(resolution = (1500, 500))
n = Observable(1)

title = @lift "α = $(@sprintf "%.2g" params.α),     Frₕ = $(@sprintf "%.2g" params.Fr_h),    Roₕ = $(@sprintf "%.2g" params.Ro_h);    " *
              "V∞ = $(@sprintf "%.2g" params.V∞) m/s,    Δz = $(@sprintf "%.2g" params.Δz_min) m,;     " *
              "Time = $(@sprintf "%s" prettytime(times[$n]))  =  $(@sprintf "%.2g" times[$n]/params.T_advective) advective periods"
fig[1, 1:length(variables)] = Label(fig, title, fontsize=18, tellwidth=false, height=title_height)

dimnames_tup = (:xF, :xC, :yF, :yC, :zF, :zC)
dimnames_dict = Dict(:xF => "x [m]", :xC => "x [m]",
                     :yF => "y [m]", :yC => "y [m]",
                     :zF => "z [m]", :zC => "z [m]",
                     :Ti => :Ti)
#---

#+++ Create axes and populate them
for (i, variable) in enumerate(variables)
    for (j, (ds, slice)) in enumerate(dslist)
        @info "Setting up $variable panel with i=$i j=$j"

        global var_raster = ds[variable]
        dimnames = collect( el for el in dimnames_tup if el in map(name, dims(var_raster)) )
        push!(dimnames, :Ti)
        @show dimnames

        v = permutedims(var_raster, dimnames)[Dim{:yC}(Between(-200, Inf))]
        vₙ = @lift v[Ti=$n]

        #+++ Set axes labels
        panel_title = j == 1              ? string(variable)      : ""
        xlabel      = j == length(dslist) ? dimnames_dict[dimnames[1]]   : ""
        ylabel      = i == 1              ? dimnames_dict[dimnames[2]]   : ""
        #---

        panel_height = slice == "xy" ? 280 : 140

        ax = Axis(fig[j+1, i], title=panel_title, xlabel=xlabel, ylabel=ylabel, height=panel_height, width=panel_width)
        global hm = heatmap!(vₙ; kwargs[variable]...)

        #+++
        if slice == "xy"
            include("bathymetry.jl")
            for (other_slice, dim, dim_value) in slicelist
                if slice == other_slice
                    global z = dim_value
                end
            end
            y = collect(dims(v, :yC))
            x_left = headland_x_of_y.(0, y, z)
            band!(Point2f.(x_left, y), Point2f.([params.Lx/2], y), color=:gray, alpha=1)
        end
        #---

        #+++ Plot vlines when appropriate
        if slice == "yz"
            vlines!(ax, params.y_south + params.sponge_length_y, color=:black, linestyle=:dash)
        end

        for (other_slice, dim, dim_value) in slicelist
            if dim == string(first(string(dimnames[1])))
                vlines!(ax, dim_value, color=:white, linestyle=:dash)
            end
        end
        #---

        #+++ Plot contours if possible
        try
            b = permutedims(ds[:b], (:xC, :zC, :Ti))
            bₙ = @lift b[:,:,$n]
            contour!(ax, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5)
        catch e
            try
                b = permutedims(ds[:b], (:yC, :zC, :Ti))
                bₙ = @lift b[:,:,$n]
                contour!(ax, bₙ; levels=10, color=:white, linestyle=:dash, linewidth=0.5)
            catch e
            end
        end
        #---
    end

    cbar_label = try metadata(var_raster)["units"] catch e "" end
    Colorbar(fig[length(dslist)+2, i], hm; label=cbar_label, vertical=false, height=cbar_height, ticklabelsize=12)
end
#---

#+++ Record animation
using DrWatson
frames = 1:step:n_times
@show step n_times max_frames length(frames)

resize_to_layout!(fig) # Resize figure after everything is done to it, but before recording
record(fig, "$(DrWatson.findproject())/anims/pres_$(simname_full).mp4", frames, framerate=14) do frame
    @info "Plotting time step $frame of $(n_times)..."
    n[] = frame
end
#---
