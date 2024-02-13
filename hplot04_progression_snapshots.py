import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from aux00_utils import open_simulation
from aux02_plotting import letterize

plt.rcParams['figure.constrained_layout.use'] = True

modifier = ""
variable = "PV_norm"
#variable = "εₖ"
Fr_h = 0.08
Ro_h = 0.2

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
                      }
#---


import matplotlib.pyplot as plt
import numpy as np


if True:
    xyz["land_mask"] = xyz["Δxᶜᶜᶜ"].where(xyz["Δxᶜᶜᶜ"] == 0)
    xyz["PV_norm"] = xyz.PV / (xyz.N2_inf * xyz.f_0)

    xyz = xyz.sel(time=np.inf, method="nearest").sel(xC=slice(-50, None), yC=slice(-250, 750))
    ds_xz = xyz.sel(yC=[0, 50, 100, 150, 200, 250,], method="nearest")
    #ds_xz = xyz.sel(yC=[0, 100, 200,], method="nearest")

    ds_xz.PV_norm.attrs = dict(long_name=r"Ertel PV / $N^2_\infty f_0$")

    fg = ds_xz[variable].pnplot(x="x", col="y", col_wrap=3,
                                cbar_kwargs = dict(shrink=0.9, fraction=0.03, pad=0.1, aspect=30, location="bottom",),
                                figsize=(15, 6), rasterized=True, **plot_kwargs_by_var[variable],
                                )
    fig = plt.gcf()

    opts_land = dict(cmap="Set2_r", vmin=0, vmax=1, alpha=1.0, zorder=10)
    for i, ax in enumerate(fg.axs.flat):
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel() + " [m]")
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel() + " [m]")
        if ax.get_title():
            ax.set_title(ax.get_title() + " m")

        ax.contour(ds_xz.xC, ds_xz.zC, ds_xz.b.pnisel(y=i), levels=15, colors="black", linestyles="--", linewidths=0.3)
        ax.pcolormesh(ds_xz.xC, ds_xz.zC, ds_xz.land_mask.pnisel(y=i), rasterized=True, **opts_land)

    #+++ Include horizontal plot on the right:
    if True:
        fig.subplots_adjust(right=0.8, bottom=0.2)
        ax = fig.add_axes((0.86, 0.1, 0.12, 0.85))

        ds_xy = xyz.sel(zC=40, method="nearest")
        ds_xy[variable].pnplot(ax=ax, x="x", add_colorbar=False, rasterized=True, **plot_kwargs_by_var[variable])

        ax.set_title("z = 40 m")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        ax.pcolormesh(ds_xy.xC, ds_xy.yC, ds_xy.land_mask, rasterized=True, **opts_land)
        for y in ds_xz.yC.values:
            ax.axhline(y=y, ls="--", lw=0.5, color="k")
    #---

    letterize(np.array([*fg.axs.flat, ax]), x=0.05, y=0.9)

else:

    #+++ Plot two panel at the same y value
    slice_name = "xiz"
    snaps = xr.open_dataset(f"data_post/{slice_name}_snaps{modifier}.nc")
    ds = snaps.isel(time=-1).pnsel(x=slice(-150, None), Fr_h=0.08, Ro_h=0.2)

    ds.PV_norm.attrs = dict(long_name=r"Ertel PV / $N^2_\infty f_0$")


    #+++
    nrows=2
    ncols=1
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                             sharex=True, sharey=True,
                             constrained_layout = True)
    axesf = axes.flatten()

    ax=axesf[0]
    ds.PV_norm.pnplot(ax=ax, x="x", vmin=-4, vmax=4, cmap=cm.balance,
                        rasterized = True,
                        cbar_kwargs = dict(label="PV/$N^2_\infty f_0$"))
    ax.set_xlabel("")

    ax=axesf[1]
    ds["εₖ"].pnplot(ax=ax, x="x", norm=LogNorm(vmin=5e-11, vmax=2e-9, clip=True), cmap="inferno",
                         rasterized=True,
                         cbar_kwargs = dict(label="KE dissipation rate [m²/s³]"),
                         )

    if nrows > 2:
        ax=axesf[2]
        ds["Ri"].pnplot(ax=ax, x="x", vmin=-1, vmax=1, cmap="RdBu_r",
                             rasterized=True,
                             cbar_kwargs = dict(label="Richardson number"),
                             )

    for ax in axesf:
        ds.b.pncontour(ax=ax, x="x", levels=20, colors="white", linestyles="--", linewidths=0.4)
        #ax.grid(True, linestyle="--", linewidth=0.4)
        ax.set_title("")
    #---

    #+++ Final touches
    Bu_h = (ds.Ro_h / ds.Fr_h)**2
    info = ", ".join((f"Roₕ = {ds.Ro_h.item():.3g}",
                      f"Frₕ = {ds.Fr_h.item():.3g}",
                      f"Buₕ = {Bu_h.item()}",
                      "y = 400 m",
                      ))
    fig.suptitle(info)
    #---
    #---

#+++ Save
fig.savefig(f"figures/{variable}_progression_{simname}.pdf", dpi=200)
#---
