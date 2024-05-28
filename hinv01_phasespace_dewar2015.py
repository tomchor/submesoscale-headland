import numpy as np
import pynanigans as pn
import xarray as xr
from aux00_utils import open_simulation
from dask.diagnostics import ProgressBar
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
            "NPN-R05F008",
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
modifiers = cycler(modifier = ["-f2"])
simnames = [ nr["name"] + nr["modifier"] for nr in modifiers * names ]
#---

for simname in simnames:
   
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
    def preprocess(ds):
        ds = ds.sel(time=slice(ds.T_advective_spinup, None))
        if "yC" in ds.coords.dims:
            ds = ds.sel(yC=slice(-ds.L, +8*ds.L))
        if "xC" in ds.coords.dims:
            ds = ds.sel(xC=slice(-ds.L, np.inf))
        ds["PV_norm"] = ds.PV / (ds.f_0 * ds.N2_inf)
        return ds


    xyz = preprocess(xyz)
    xiz = preprocess(xiz)
    xyi = preprocess(xyi)
    tti = preprocess(tti)

    tti = tti.mean("time")
    #---

    #+++ Plot context plots
    from matplotlib import pyplot as plt
    x0 = 100
    opts = dict(x="x", vmin=-2, vmax=2, cmap=plt.cm.RdBu_r)
    tti.PV_norm.pnplot(**opts)
    plt.scatter(x0, xiz.yC.item(), color="k", s=100)

    xyi.PV_norm.sel(time = slice(None, None, 30)).pnplot(col="time", **opts)
    for ax in plt.gcf().axes:
        ax.scatter(x0, xiz.yC.item(), color="k", s=100)
    #---

    #+++ Look for waves
    fig, axes = plt.subplots(nrows=2, figsize=(10, 5), sharex=True)

    xiz = xiz.assign_coords(time = xiz.time * xiz.T_advective / xiz.T_inertial)
    xiz.time.attrs = dict(units = "Inertial periods")

    xiz["∂u∂z"].sel(xC=x0, method="nearest").pnplot(ax=axes[0], x="time")
    xiz["∂v∂z"].sel(xC=x0, method="nearest").pnplot(ax=axes[1], x="time")

    for ax in fig.axes:
        ax.set_title("")
    #---
    pause
