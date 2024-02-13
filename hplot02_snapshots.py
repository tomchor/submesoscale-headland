import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
sys.path.append("/glade/u/home/tomasc/repos/xanimations")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from aux02_plotting import manual_facetgrid, get_orientation
from cmocean import cm
plt.rcParams["figure.constrained_layout.use"] = True
π = np.pi

modifier = ""
slice_name = "etfields"
#slice_name = "xyi"
#slice_name = "xiz"
#slice_name = "iyz"

#+++ Read and reindex dataset
snaps = xr.open_dataset(f"data_post/{slice_name}_snaps{modifier}.nc")
snaps = snaps.reindex(Ro_h = list(reversed(snaps.Ro_h)))
if 1:
    snaps = snaps.isel(time=-1).sel(λ=100)
else:
    snaps = snaps.chunk(time="auto").sel(time=slice(None, None, 1)).mean("time", keep_attrs=True).sel(λ=100)
    snaps = snaps.expand_dims(time = [0]).isel(time=0)

try:
    snaps = snaps.reset_coords(("zC", "zF"))
except ValueError:
    pass
#---

#+++ Adjust/create variables
snaps.PV_norm.attrs = dict(long_name=r"Normalized Ertel PV")

snaps["Lo"] = 2*π * np.sqrt(snaps["εₖ"]  / snaps["N²∞"]**(3/2))
snaps["Δz_norm"] = snaps.Δz_min / snaps["Lo"]
snaps["Δz_norm"].attrs = dict(long_name="Δz / Ozmidov scale")

if "q̃" in snaps.variables.keys():
    snaps["q̃_norm"] = snaps["q̃"]  / (snaps["N²∞"] * snaps["f₀"])
    snaps["q̃_norm"].attrs = dict(long_name=r"Normalized filtered Ertel PV")

if "wb" in snaps.variables.keys():
    snaps["Kb"] = -snaps.wb / snaps["N²∞"]

snaps["γ"] = snaps["εₚ"] / (snaps["εₚ"] + snaps["εₖ"])

Π_thres = 1e-11
Π_std = snaps["Π"].where(snaps.water_mask).pnstd(("x", "y"))
snaps["R_Π"]  = (snaps["Πᵃ"]          / Π_std)#.where((snaps["Πᵃ"] > 0)          & (snaps["Π"] > Π_thres))
snaps["R_SP"] = (snaps["SP"].sel(j=3) / Π_std).where((snaps["SP"].sel(j=3) > 0))
#---

#+++ Options
sel = dict()
if "xC" in snaps.coords: # has an x dimension
    sel = sel | dict(x=slice(-snaps.headland_intrusion_size_max/3, np.inf))
if "yC" in snaps.coords: # has a y dimension
    sel = sel | dict(y=slice(-snaps.runway_length/2, np.inf))
if "zC" in snaps.coords: # has a z dimension
    sel = sel | dict(z=slice(None))

cbar_kwargs = dict(shrink=0.5, fraction=0.012, pad=0.02, aspect=30)
if ("xC" in snaps.coords) and ("yC" in snaps.coords):
    figsize = (8, 8)
    cbar_kwargs = dict(location="right") | cbar_kwargs
else:
    figsize = (9, 6)
    cbar_kwargs = dict(location="bottom") | cbar_kwargs

opts_orientation = get_orientation(snaps)
#---

#+++ Variables plot_kwargs
plot_kwargs_by_var = {#"PV_norm"   : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      #"q̃_norm"    : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      #"PVz_ratio" : dict(vmin=-10, vmax=10, cmap="RdBu_r"),
                      #"PVh_ratio" : dict(vmin=-10, vmax=10, cmap="RdBu_r"),
                      #"Ri"        : dict(vmin=-2, vmax=2, cmap=cm.balance),
                      #"Ro"        : dict(vmin=-3, vmax=3, cmap="bwr"),
                      #"εₖ"        : dict(norm=LogNorm(vmin=1e-10, vmax=1e-8, clip=True), cmap="inferno"),
                      #"εₚ"        : dict(norm=LogNorm(vmin=1e-10, vmax=1e-8, clip=True), cmap="inferno"),
                      #"Lo"        : dict(vmin=0, vmax=2, cmap=cm.balance),
                      #"Δz_norm"   : dict(vmin=0, vmax=2, cmap=cm.balance),
                      #"v"         : dict(vmin=-1.2*snaps.V_inf, vmax=1.2*snaps.V_inf, cmap=cm.balance),
                      #"wb"        : dict(vmin=-1e-8, vmax=+1e-8, cmap=cm.balance),
                      #"Kb"        : dict(vmin=-1e-1, vmax=+1e-1, cmap=cm.balance),
                      #"γ"         : dict(vmin=0, vmax=1, cmap="plasma"),
                      "Π"         : dict(cmap=cm.balance, robust=True),
                      "R_Π"       : dict(cmap=cm.balance, robust=True),
                      #"R_SP"      : dict(cmap=cm.balance, robust=True),
                      }
#---

for var, opts in plot_kwargs_by_var.items():
    if var not in snaps.variables.keys():
        print(f"Skipping {slice_name} slices of {var} since it doesn't seem to be in the file.")
        continue
    print(f"Plotting {slice_name} slices of {var} with modifier={modifier}")
    fig = plt.figure(figsize=figsize)
    aux, fig = manual_facetgrid(snaps[var].pnsel(**sel), fig,
                                land_mask = snaps.land_mask.biject(),
                                plot_kwargs = (opts | opts_orientation),
                                cbar_kwargs = cbar_kwargs,
                                )

    #+++ Final touches and save
    #info = ", ".join((f"Δzₘᵢₙ = {snaps.Δz_min.min().item():.3g} m",
    #                  f"time = {snaps.time.item():.3g} cycle periods",
    #                  f"α = {snaps.α:.3g}",
    #                  f"V∞ = {snaps.V_inf:.3g} m/s",
    #                  ))
    #fig.suptitle(info)
    fig.suptitle("")
    fig.savefig(f"figures_check/{var}_{slice_name}{modifier}.pdf", dpi=200)
    #---
