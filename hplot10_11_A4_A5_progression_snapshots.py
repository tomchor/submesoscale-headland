import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from aux00_utils import open_simulation
from aux02_plotting import letterize, plot_kwargs_by_var
plt.rcParams["font.size"] = 9

modifiers = ["", "-S"]
variable_xy = "PV_norm"
variables = ["PV_norm", "εₖ", "ω_y"]
Fr_h = 0.2
Ro_h = 1 # Change this value to 0.2 to reproduce Figures A4 and A5

#+++ Pick downstream distances
N = 3
if (Fr_h==0.08) and (Ro_h==1):
    downstream_distances = np.linspace(0, 200, N)
elif (Fr_h==0.08) and (Ro_h==0.08):
    downstream_distances = np.linspace(0, 200, N)
elif (Fr_h==0.2) and (Ro_h==0.2):
    downstream_distances = np.linspace(80, 300, N)
elif (Fr_h==0.2) and (Ro_h==0.5):
    downstream_distances = np.linspace(0, 300, N)
elif (Fr_h==0.2) and (Ro_h==1):
    downstream_distances = np.linspace(80, 300, N)
else:
    downstream_distances = np.linspace(0, 200, N)
#---


plot_kwargs_by_var["PV_norm"] = dict(vmin=-10, vmax=+10, cmap=plt.cm.RdBu_r)
plot_kwargs_by_var["εₖ"] = dict(norm=LogNorm(vmin=1e-10/3,   vmax=1e-8/3,   clip=True), cmap="inferno")
plot_kwargs_by_var["ω_y"] = dict(vmin=-3e-3, vmax=3e-3, cmap=cm.balance)

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

    #+++ Adjust attributes
    xyz["land_mask"] = xyz["Δxᶜᶜᶜ"].where(xyz["Δxᶜᶜᶜ"] == 0)
    xyz["PV_norm"] = xyz.PV / (xyz.N2_inf * xyz.f_0)

    xyz = xyz.sel(time=np.inf, method="nearest").sel(xC=slice(-200, None), yC=slice(-200, 1000))
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
    size = 2.
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(2*size*ncols+4, size*nrows),
                             sharex=True, sharey=True)
    #---

    #+++ Plot stuff
    pcms = []
    for i, yC in enumerate(ds_xz.yC.values):
        print(f"y = {yC} m")
        for j, variable in enumerate(variables):
            ax = axes[j, i]
            pcm = ds_xz[variable].sel(yC=yC).pnplot(ax=ax, x="x", **plot_kwargs_by_var[variable], add_colorbar=False, rasterized=True)
            pcms.append(pcm)
    #---

    #+++ Plot bathymetry and adjust panels
    print("Plotting bathymetry")
    opts_land = dict(cmap="Set2_r", vmin=0, vmax=1, alpha=1.0, zorder=10)
    for i, ax_col in enumerate(axes.T):
        ax_col[0].set_title(f"y = {ds_xz.yC.values[i]:.1f} m")
    
        for j, ax in enumerate(ax_col):
            if j != 1:
                ax.contour(ds_xz.xC, ds_xz.zC, ds_xz.b.pnisel(y=i), levels=15, colors="black", linestyles="--", linewidths=0.3, rasterized=True)
            ax.pcolormesh(ds_xz.xC, ds_xz.zC, ds_xz.land_mask.pnisel(y=i), rasterized=True, **opts_land)
            if i>0:
                ax.set_ylabel("")
            if j>0:
                ax.set_title(f"")
            if j<len(variables)-1:
                ax.set_xlabel("")
    fig.tight_layout(h_pad=0)
    #---

    #+++ Include horizontal plot on the right:
    print("Including horizontal plot to the right")
    fig.subplots_adjust(right=0.8,)
    ax = fig.add_axes((0.845, 0.1, 0.15, 0.85))

    ds_xy = xyz.sel(zC=40, method="nearest")
    ds_xy[variable_xy].pnplot(ax=ax, x="x", add_colorbar=False, rasterized=True, **plot_kwargs_by_var[variable_xy])

    ax.set_title(f"z = {ds_xy.zC.values:.2f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    ax.pcolormesh(ds_xy.xC, ds_xy.yC, ds_xy.land_mask, rasterized=True, **opts_land)
    for y in ds_xz.yC.values:
        ax.axhline(y=y, ls="-", lw=0.5, color="k")

    for i in range(len(variables)):
        cax = axes[i,0].inset_axes([200, 0, 200, 5], transform=axes[i,0].transData, zorder=100)
        c_ax = fig.colorbar(pcms[i], cax=cax, orientation="horizontal", label=ds_xz[variables[i]].attrs["long_name"], location="top")
    #---

    #+++ Save
    print("saving...")
    letterize(np.array([*axes.flatten(), ax]), x=0.05, y=0.9)
    fig.savefig(f"figures/{variable_xy}_progression_{simname}.pdf", dpi=200)
    print()
    #---
