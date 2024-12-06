import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from aux00_utils import open_simulation
from aux01_physfuncs import calculate_filtered_PV
from aux02_plotting import letterize

plt.rcParams['figure.constrained_layout.use'] = True

modifier = "-f4"
Fr_h = 0.2
Ro_h = 1

#+++ Open dataset
path = f"./headland_simulations/data/"
simname = f"NPN-R%sF%s{modifier}" % (str(Ro_h).replace(".", ""), str(Fr_h).replace(".", ""))
grid_xyz, xyz = open_simulation(path + f"xyz.{simname}.nc",
                                use_advective_periods=True,
                                topology="NPN",
                                squeeze=True,
                                load=False,
                                open_dataset_kwargs=dict(chunks=dict(time=1)),
                                )
#---

#+++ Variables plot_kwargs
plot_kwargs_by_var = {"PV_norm"   : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
                      "q̃_norm"    : dict(vmin=-6, vmax=6, cmap="RdBu_r"),
                      "ω_y"       : dict(vmin=-2e-3, vmax=+2e-3, cmap=cm.balance),
                      "PVz_ratio" : dict(vmin=-10, vmax=10, cmap="RdBu_r"),
                      "PVh_ratio" : dict(vmin=-10, vmax=10, cmap="RdBu_r"),
                      "Ri"        : dict(vmin=-2, vmax=2, cmap=cm.balance),
                      "Ro"        : dict(vmin=-3, vmax=3, cmap="bwr"),
                      "εₖ"        : dict(norm=LogNorm(vmin=1e-10, vmax=1e-8, clip=True), cmap="inferno"),
                      "εₚ"        : dict(norm=LogNorm(vmin=1e-10, vmax=1e-8, clip=True), cmap="inferno"),
                      "Lo"        : dict(vmin=0, vmax=2, cmap=cm.balance),
                      "Δz_norm"   : dict(vmin=0, vmax=2, cmap=cm.balance),
                      "v"         : dict(vmin=-1.2*xyz.V_inf, vmax=1.2*xyz.V_inf, cmap=cm.balance),
                      "wb"        : dict(vmin=-1e-8, vmax=+1e-8, cmap=cm.balance),
                      "Kb"        : dict(vmin=-1e-1, vmax=+1e-1, cmap=cm.balance),
                      }
#---


import matplotlib.pyplot as plt
import numpy as np

xyz["f₀"] = xyz.f_0
xyz = calculate_filtered_PV(xyz, scale_meters = 5, condense_tensors=True, indices = [1,2,3], cleanup=False)
xyz["land_mask"] = xyz["Δxᶜᶜᶜ"].where(xyz["Δxᶜᶜᶜ"] == 0)
xyz["PV_norm"] = xyz.PV / (xyz.N2_inf * xyz.f_0)
xyz["q̃_norm"] = xyz.q̃ / (xyz.N2_inf * xyz.f_0)

xyz = xyz.sel(time=np.inf, method="nearest").sel(xC=slice(-50, None), yC=slice(-250, 750))
ds_xz = xyz.sel(yC=[100, 250, 500], method="nearest")

ds_xz.PV_norm.attrs = dict(long_name=r"Ertel PV / $N^2_\infty f_0$")
ds_xz.q̃_norm.attrs = dict(long_name=r"Filtered Ertel PV / $N^2_\infty f_0$")
ds_xz["εₖ"].attrs = dict(long_name=r"KE dissipation rate [m²/s³]")
ds_xz["ω_y"].attrs = dict(long_name=r"Streamwise vorticity [1/s]")

size = 2.5
nrows=2; ncols=len(ds_xz.yC)
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3*size*ncols, size*nrows),
                         squeeze=False,
                         sharex=True, sharey=True)

opts_land = dict(cmap="Set2_r", vmin=0, vmax=1, alpha=1.0, zorder=10)
cbar_kwargs_fixed = dict(shrink=0.9, fraction=0.03, pad=0, aspect=30, location="right",)
for i, yC in enumerate(ds_xz.yC.values):
    print(f"Plotting {yC}")
    if yC == ds_xz.yC[-1]:
        add_colorbar = True
        cbar_kwargs = cbar_kwargs_fixed
    else:
        add_colorbar = False
        cbar_kwargs = None
    ax = axes[0,i]
    ds_xz.sel(yC=yC)["q̃_norm"].pnplot(ax=ax, x="x", rasterized=True, cbar_kwargs=cbar_kwargs, add_colorbar=add_colorbar, **plot_kwargs_by_var["PV_norm"])
    ax.pcolormesh(ds_xz.xC, ds_xz.zC, ds_xz.land_mask.pnisel(y=i), rasterized=True, **opts_land)
    ax.set_xlabel("")
    ax.set_ylabel("z [m]")
    ax.set_title(f"y = {yC:.2f}")

    ax = axes[1,i]
    ds_xz.sel(yC=yC)["ω_y"].pnplot(ax=ax, x="x", rasterized=True, cbar_kwargs=cbar_kwargs, add_colorbar=add_colorbar, **plot_kwargs_by_var["ω_y"])
    ax.pcolormesh(ds_xz.xC, ds_xz.zC, ds_xz.land_mask.pnisel(y=i), rasterized=True, **opts_land)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("")

#+++ Save
fig.savefig(f"figures/CSI_progression_{simname}.pdf", dpi=200)
#---
