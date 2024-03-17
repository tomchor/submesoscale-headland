using Rasters
import NCDatasets

if !(@isdefined simname) || (typeof(simname) !== String)
    simname = "NPN-R1F008-f4"
end

@show "Reading NetCDF"
xyz = RasterStack("data/xyz.$simname.nc", name=(:PV,), lazy=true)

md = metadata(xyz)
params = (; (Symbol(k) => v for (k, v) in md)...)

#+++ Define headland as x(y, z)
@inline η(z, p) = 2*p.L + (0 - 2*p.L) * z / (2*p.H) # headland intrusion size
@inline headland_width(z, p) = p.β * η(z, p)
@inline headland_x_of_yz(y, z, p) = 2*p.L - η(z, p) * exp(-(2y / headland_width(z, p))^2)
@inline headland_continuous(x, y, z) = headland_x_of_yz(y, z, params) - x
@inline bathymetry(x, y, z) = z > 0 ? headland_continuous(x, y, z) : 0
#---

@show "Slicing xyz"
xyz = xyz[yC=Between(-md["runway_length"], Inf), xC=Between(dims(PV, :xC)[2], Inf)]
xC = Array(dims(xyz, :xC))
yC = Array(dims(xyz, :yC))
zC = Array(dims(xyz, :zC))

PV_lim = 1.5 + params.Ro_h
PV_lims = (-PV_lim, +PV_lim)
H = [ bathymetry(x, y, z) < 5 ? 1 : 0 for x=xC, y=yC, z=zC ]

using GLMakie
fig = Figure(resolution = (1200, 900));
n = Observable(1)

@show "Slicing PV"
PV = xyz.PV[Ti=Between(params.T_advective_spinup * params.T_advective, Inf)] ./ (md["N2_inf"] * md["f_0"])
PVₙ = @lift Array(PV)[:,:,:,$n]

colormap = to_colormap(:balance)
middle_chunk = ceil(Int, 1.5 * 128 / PV_lim) # Needs to be *at least* larger than 128 / PV_lim
colormap[128-middle_chunk:128+middle_chunk] .= RGBAf(0,0,0,0)

settings_axis3 = (aspect = (md["Lx"], md["Ly"], 4*md["Lz"]), azimuth = -0.80π, elevation = 0.2π,
                  perspectiveness=0.8, viewmode=:fitzoom, xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")

ax = Axis3(fig[2, 1]; settings_axis3...)
volume!(ax, xC, yC, zC, H, algorithm = :absorption, absorption=50f0, colormap = [:papayawhip, RGBAf(0,0,0,0), :papayawhip], colorrange=(-1, 1)) # turn on anti-aliasing

vol = volume!(ax, xC, yC, zC, PVₙ, algorithm = :absorption, absorption=20f0, colormap=colormap, colorrange=PV_lims)
Colorbar(fig, vol, bbox=ax.scene.px_area,
         label="PV / N²∞ f₀", height=25, width=Relative(0.35), vertical=false,
         alignmode = Outside(10), halign = 0.15, valign = 0.02)


#+++ Inset axis
inset_ax = Axis3(fig[2, 1]; width=Relative(0.4), height=Relative(0.4), halign=1, valign=1.1, zticks=[0, 40, 80], settings_axis3...)
contour!(inset_ax, xC, yC, zC, bathymetry, levels = [0], specular = Vec3f(0), diffuse = Vec3f(1), colormap = [:gray50], fxaa = true,) # turn on anti-aliasing

ps = [Point3f(0, -400, 40)]
ns = [Vec3f(0, params.Ly/2, 0)]
arrows!(inset_ax, ps, ns,
        fxaa = true, # turn on anti-aliasing
        color = :brown2, linewidth = 12, arrowsize = Vec3f(100, 20, 100))

text!(Point3f(-200, -100 + params.Ly/2, 40),
      text = "V∞ = 1 cm/s", rotation = 1.5π, align = (:left, :baseline),
      fontsize = 200, markerspace = :data, color = :brown2)
#---

#+++ Save a snapshot as png
n[] = length(dims(PV, :Ti))
save(string(@__DIR__) * "/../../figures/bathymetry_3d_PV_$simname.png", fig, px_per_unit=2);
#---

#+++ Define title with time
using Printf
using Oceananigans: prettytime
title = @lift "Frₕ = $(@sprintf "%.2g" params.Fr_h),    Roₕ = $(@sprintf "%.2g" params.Ro_h);    " *
              "Time = $(@sprintf "%s" prettytime(dims(PV, :Ti)[$n]))"
fig[1, 1] = Label(fig, title, fontsize=18, tellwidth=false, height=8)
#---

pause
#+++ Record animation
n[] = 1
frames = 1:length(dims(PV, :Ti))
GLMakie.record(fig, string(@__DIR__) * "/../../anims/bathymetry_3d_PV_$simname.mp4", frames, framerate=14) do frame
    @info "Plotting time step $frame"
    n[] = frame
end
#---
