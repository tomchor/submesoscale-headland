import numpy as np
import pynanigans as pn
import xarray as xr
from aux00_utils import open_simulation
from aux02_plotting import plot_kwargs_by_var
from matplotlib.colors import LogNorm
π = np.pi

#+++ Define dir and file names
path = f"./headland_simulations/data/"
simnames = [#"NPN-TEST",
            #"NPN-R008F008",
            #"NPN-R008F02",
            #"NPN-R008F05",
            #"NPN-R008F1",
            "NPN-R02F008",
            #"NPN-R02F02",
            #"NPN-R02F05",
            #"NPN-R02F1",
            #"NPN-R05F008",
            #"NPN-R05F02",
            #"NPN-R05F05",
            #"NPN-R05F1",
            #"NPN-R1F008",
            #"NPN-R1F02",
            #"NPN-R1F05",
            #"NPN-R1F1",
            ]

from cycler import cycler
names = cycler(name=simnames)
modifiers = cycler(modifier = [""])
simnames = [ nr["name"] + nr["modifier"] for nr in modifiers * names ]
#---

for simname in simnames:
    print("\n", simname)
   
    #+++ Open datasets xyz and xyi
    print(f"\nOpening {simname} xyz")
    grid_xyz, xyz = open_simulation(path+f"xyz.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks="auto"),
                                    )
    print(f"Opening {simname} xiz")
    grid_xiz, xiz = open_simulation(path+f"xiz.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks="auto"),
                                    )
    print(f"Opening {simname} xyi")
    grid_xyi, xyi = open_simulation(path+f"xyi.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks="auto"),
                                    )
    print(f"Opening {simname} tti")
    grid_tti, tti = open_simulation(path+f"tti.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks="auto"),
                                    )
    #---

    #+++ Pre-process data
    def preprocess(ds, ϵ=0):
        ds = ds.sel(time=slice(ds.T_advective_spinup+ϵ, None))
        if "yC" in ds.coords.dims:
            ds = ds.sel(yC=slice(-ds.L, +8*ds.L))
        if "xC" in ds.coords.dims:
            ds = ds.sel(xC=slice(-ds.L, np.inf))
        ds["PV_norm"] = ds.PV / (ds.f_0 * ds.N2_inf)
        return ds


    xyz = preprocess(xyz)
    xiz = preprocess(xiz)
    xyi = preprocess(xyi)
    tti = preprocess(tti, ϵ=1e-3)

    tti = tti.mean("time")

    xiz["dUdz"] = np.sqrt(xiz["∂u∂z"]**2 + xiz["∂v∂z"]**2)
    dudz_opts = dict(x="time", vmin=-2e-3, vmax=2e-3, cmap=plt.cm.RdBu_r)
    #---

    #+++ Plot time-avg PV
    from matplotlib import pyplot as plt
    plt.rcParams["figure.constrained_layout.use"] = True

    if (np.round(xyz.Ro_h, decimals=2) == 0.2) and (np.round(xyz.Fr_h, decimals=2) == 0.2):
        x0 = 250
    else:
        x0 = 100
    title = f"Roₕ = {xyz.Ro_h.item():.2f}, Frₕ = {xyz.Fr_h.item():.2f}"

    PV_opts = dict(vmin=-2, vmax=2, cmap=plt.cm.RdBu_r)
    ε_opts = dict(norm=LogNorm(vmin=1e-13, vmax=1e-10, clip=True), cmap="inferno")
    if 0:
        plt.figure()
        tti.PV_norm.pnplot(x="x", **PV_opts)
        plt.scatter(x0, xiz.yC.item(), color="k", s=100)
        plt.title(title)
        #---

    #+++ Plot snapshots
    times = np.arange(xyi.time[0], xyi.time[-1]+1, 10)
    if 1:
        xyi.PV_norm.sel(time=times, method="nearest").pnplot(col="time", x="x", **PV_opts)
        plt.suptitle(title)
        for ax in plt.gcf().axes:
            ax.scatter(x0, xiz.yC.item(), color="k", s=100)
    #---

    #+++ Look for subinertial waves with a Hovmoeller diagram
    if 0:
        fig, axes = plt.subplots(nrows=3, figsize=(12, 8), sharex=True)

        xiz = xiz.assign_coords(time = xiz.time * xiz.T_advective / xiz.T_inertial)
        xiz.time.attrs = dict(units = "Inertial periods")

        xiz["∂u∂z"].sel(xC=x0, method="nearest").pnplot(ax=axes[0], **dudz_opts)
        xiz["∂v∂z"].sel(xC=x0, method="nearest").pnplot(ax=axes[1], **dudz_opts)
        xiz["PV_norm"].sel(xC=x0, method="nearest").pnplot(ax=axes[2], x="time", **PV_opts)

        fig.suptitle(title)
        for ax in axes:
            ax.set_xticks(np.arange(np.ceil(xiz.time[0]), np.round(xiz.time[-1])+1, 2), minor=False)
            ax.set_xticks(np.arange(np.round(xiz.time[0]), np.round(xiz.time[-1])+1, 1), minor=True)
            ax.grid(which="both", axis="both")
            ax.set_title("")
    #---

    #+++ Check out vertical snapshots
    if 0:
        fig, axes = plt.subplots(nrows=len(times), ncols=2, figsize=(12, 12), sharex=True, sharey=True)
        for i, time in enumerate(times):
            row = axes[i]
            xiz_row = xiz.sel(time=time, method="nearest")

            xiz_row["PV_norm"].pnplot(ax=row[0], x="x", add_colorbar=False, **PV_opts)
            xiz_row["εₖ"].pnplot(ax=row[1], x="x", **ε_opts)

        for ax in axes.flatten():
            ax.set_xlabel("")
            ax.set_title("")
        for ax in axes[:,1]:
            ax.set_ylabel("")
        for ax, time in zip(axes[:,0], times):
            ax.set_ylabel(f"time = {time}")
    #---
