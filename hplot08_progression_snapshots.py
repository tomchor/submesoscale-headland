import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from aux00_utils import open_simulation
from aux02_plotting import letterize

modifiers = ["", "-S"]
variable_xy = "PV_norm"
variables = ["PV_norm", "εₖ"]
Fr_h = 0.2
Ro_h = 0.5

#+++ Pick downstream distances
if (Fr_h==0.08) and (Ro_h==1):
    downstream_distances = [0, 100, 200,]
elif (Fr_h==0.08) and (Ro_h==0.08):
    downstream_distances = [0, 50, 100,]
elif (Fr_h==0.2) and (Ro_h==0.2):
    downstream_distances = [0, 50, 100,]
elif (Fr_h==0.2) and (Ro_h==0.5):
    downstream_distances = [0, 75, 150,]
elif (Fr_h==0.2) and (Ro_h==1):
    downstream_distances = [0, 100, 200,]
else:
    downstream_distances = [0, 50, 100,]
#---

for modifier in modifiers:
    print(f"Opening modifier={modifier}")

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
                          "q̃_norm"    : dict(vmin=-5, vmax=5, cmap="RdBu_r"),
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
                          "ω_y"       : dict(vmin=-3e-3, vmax=+3e-3, cmap=cm.balance),
                          }
    #---

    #+++ Adjust attributes
    xyz["land_mask"] = xyz["Δxᶜᶜᶜ"].where(xyz["Δxᶜᶜᶜ"] == 0)
    xyz["PV_norm"] = xyz.PV / (xyz.N2_inf * xyz.f_0)

    xyz = xyz.sel(time=np.inf, method="nearest").sel(xC=slice(-50, None), yC=slice(-250, 750))
    ds_xz = xyz.sel(yC=downstream_distances, method="nearest")

    ds_xz.PV_norm.attrs = dict(long_name=r"Ertel PV / $N^2_\infty f_0$")
    ds_xz["εₖ"].attrs = dict(long_name=r"$\varepsilon_k$ [m²/s³]")
    ds_xz["ω_y"].attrs = dict(long_name=r"$y$-vorticity [1/s]")
    ds_xz.xC.attrs["long_name"] = "$x$"
    ds_xz.zC.attrs["long_name"] = "$z$"
    #---

    #+++ Create figure
    ncols = len(ds_xz.yC)
    nrows = len(variables)
    size = 2.5
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(1.5*size*ncols+4, size*nrows),
                             sharex=True, sharey=True)
    #---

    #+++ Plot stuff
    pcms = []
    for i, yC in enumerate(ds_xz.yC.values):
        for j, variable in enumerate(variables):
            ax = axes[j, i]
            pcm = ds_xz[variable].sel(yC=yC).pnplot(ax=ax, x="x", **plot_kwargs_by_var[variable], add_colorbar=False, rasterized=True)
            pcms.append(pcm)
    #---

    #+++ Plot bathymetry and adjust panels
    opts_land = dict(cmap="Set2_r", vmin=0, vmax=1, alpha=1.0, zorder=10)
    for i, ax_col in enumerate(axes.T):
        ax_col[0].set_title(f"{ds_xz.yC.values[i]:.2f} m")
        ax_col[1].set_title(f"")
    
        ax_col[0].set_xlabel("")
        ax_col[0].contour(ds_xz.xC, ds_xz.zC, ds_xz.b.pnisel(y=i), levels=15, colors="black", linestyles="--", linewidths=0.3)
        for ax in ax_col:
            ax.pcolormesh(ds_xz.xC, ds_xz.zC, ds_xz.land_mask.pnisel(y=i), rasterized=True, **opts_land)
            if i>0:
                ax.set_ylabel("")
    fig.tight_layout(h_pad=0)
    #---

    #+++ Include horizontal plot on the right:
    if True:
        fig.subplots_adjust(right=0.8,)
        ax = fig.add_axes((0.86, 0.1, 0.12, 0.85))

        ds_xy = xyz.sel(zC=40, method="nearest")
        ds_xy[variable_xy].pnplot(ax=ax, x="x", add_colorbar=False, rasterized=True, **plot_kwargs_by_var[variable_xy])

        ax.set_title(f"z = {ds_xy.zC.values:.2f} m")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        ax.pcolormesh(ds_xy.xC, ds_xy.yC, ds_xy.land_mask, rasterized=True, **opts_land)
        for y in ds_xz.yC.values:
            ax.axhline(y=y, ls="--", lw=0.5, color="k")

        for i in range(len(variables)):
            cax = axes[i,0].inset_axes([150, 5, 200, 5], transform=axes[i,0].transData, zorder=100)
            c_ax = fig.colorbar(pcms[i], cax=cax, orientation="horizontal", label=ds_xz[variables[i]].attrs["long_name"], location="top")
    #---

    #+++ Save
    print("saving...")
    letterize(np.array([*axes.flatten(), ax]), x=0.05, y=0.9)
    fig.savefig(f"figures/{variable_xy}_progression_{simname}.pdf", dpi=200)
    print()
    #---
