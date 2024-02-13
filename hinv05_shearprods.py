import numpy as np
import pynanigans as pn
import xarray as xr
from aux00_utils import filter_by_resolution, all_wake_sims
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
π = np.pi


#+++ Define directory and simulation name
path = f"./headland_simulations/data/"
simnames = [#"NPN-TEST-f8",
            #"NPN-PropA-f8",
            #"NPN-PropA-f4",
            #"NPN-PropA-f2",
            #"NPN-PropA",
            #"NPN-PropB-f8",
            #"NPN-PropB-f4",
            #"NPN-PropB-f2",
            #"NPN-PropB",
            #"NPN-PropD-f8",
            #"NPN-PropD-f4",
            #"NPN-PropD-f2",
            #"NPN-PropD",
            "NPN-R008F008-f8",
            "NPN-R008F008-f4",
            "NPN-R008F008-f2",
            "NPN-R008F008",
            "NPN-R008F02-f8",
            "NPN-R008F02-f4",
            "NPN-R008F02-f2",
            "NPN-R008F02",
            "NPN-R008F05-f8",
            "NPN-R008F05-f4",
            "NPN-R008F05-f2",
            "NPN-R008F05",
            "NPN-R008F1-f8",
            "NPN-R008F1-f4",
            "NPN-R008F1-f2",
            "NPN-R008F1",
            "NPN-R02F008-f8",
            "NPN-R02F008-f4",
            "NPN-R02F008-f2",
            "NPN-R02F008",
            "NPN-R02F02-f8",
            "NPN-R02F02-f4",
            "NPN-R02F02-f2",
            "NPN-R02F02",
            "NPN-R02F05-f8",
            "NPN-R02F05-f4",
            "NPN-R02F05-f2",
            "NPN-R02F05",
            "NPN-R02F1-f8",
            "NPN-R02F1-f4",
            "NPN-R02F1-f2",
            "NPN-R02F1",
            "NPN-R05F008-f8",
            "NPN-R05F008-f4",
            "NPN-R05F008-f2",
            "NPN-R05F008",
            "NPN-R05F02-f8",
            "NPN-R05F02-f4",
            "NPN-R05F02-f2",
            "NPN-R05F02",
            "NPN-R05F05-f8",
            "NPN-R05F05-f4",
            "NPN-R05F05-f2",
            "NPN-R05F05",
            "NPN-R05F1-f8",
            "NPN-R05F1-f4",
            "NPN-R05F1-f2",
            "NPN-R05F1",
            "NPN-R1F008-f8",
            "NPN-R1F008-f4",
            "NPN-R1F008-f2",
            "NPN-R1F008",
            "NPN-R1F02-f8",
            "NPN-R1F02-f4",
            "NPN-R1F02-f2",
            "NPN-R1F02",
            "NPN-R1F05-f8",
            "NPN-R1F05-f4",
            "NPN-R1F05-f2",
            "NPN-R1F05",
            "NPN-R1F1-f8",
            "NPN-R1F1-f4",
            "NPN-R1F1-f2",
            "NPN-R1F1",
            ]
#---

n_cycles = 5
λ_approx = [20, 100, 200, 400]
resolutions = ["f2", ""]
for resolution in resolutions:
    simnames = all_wake_sims(simnames)
    simnames_filtered = [ simname for simname in simnames if filter_by_resolution(simname, resolution) ]

    #+++ Create figure and options
    mosaic = np.vstack(([-1, -1], np.arange(0, 4).reshape(2,2)))
    fig, axd = plt.subplot_mosaic(mosaic, layout="constrained", figsize=(11, 12))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
    #---

    dslist = []
    for sim_number, simname in enumerate(simnames_filtered):

        #+++ Load fields dataset
        print(f"Opening {simname}")
        fields = xr.open_dataset(f"data_post/fields_{simname}.nc", chunks=dict(time="auto", λ="auto"))
        fields = fields[["Πᵍ", "Πᵃ", "Π", "land_mask_buffered_filt", "Ek̃"]]
        #---

        #+++ Filter and preprocess
        t_slice = slice(fields.time[-1] - n_cycles, np.inf)
        fields = fields.sel(time=t_slice)
        
        mfields = fields[["Πᵍ", "Πᵃ", "Π", "Ek̃"]].where(np.logical_not(fields.land_mask_buffered_filt).compute(), drop=True)
        #mfields_xyavg = mfields.sel(method="nearest").mean(("x", "y"))
        mfields_xyavg = mfields.sel(method="nearest").mean(("x", "y", "time")).expand_dims("time")
        #---

        #+++ Plot GSP vs AGSP
        for i, λ in enumerate(λ_approx):
            ax = axd[i]
            CSI_mean_L = mfields_xyavg.sel(λ=λ, method="nearest")
            CSI_mean_L.plot.scatter(ax=ax, x="Πᵍ", y="Πᵃ",
                                    #s=np.sqrt(abs(CSI_mean_L["Π"]))*1e7,
                                    edgecolors="none", label=simname[4:])
            ax.set_title(f"Filter scale = {CSI_mean_L.λ.item():.3f} m")
        #---

        #+++ Plot Energy spectrum
        mfields_txyavg = mfields_xyavg.mean("time")
        #---

        #+++ Plot total energy transfer spectrum
        mfields_txyavg["Π_norm"] = mfields_txyavg.Π / abs(mfields_txyavg.Π).max()
        mfields_txyavg.Π_norm.plot(ax=axd[-1], xscale="log", label=simname[4:])
        #---

    print("Adjusting panels")

    #+++ Adjust energy flux panel
    ax = axd[-1]
    ax.grid(True)
    ax.set_title("Π / max(abs(Π))")
    ax.legend(loc="upper right", bbox_to_anchor=(-.11, 1), ncol=1)
    ax.axhline(y=0, ls="--", color="k", zorder=-1)
    #---

    #+++ Adjust scatterplot panels
    lim = 1e-9
    for i in range(0, max(axd.keys())+1):
        ax = axd[i]
        ax.grid(True)
        ax.set_xlim(-lim, +lim)
        ax.set_ylim(-lim, +lim)
        ax.set_xscale("symlog", linthresh=lim/1000)
        ax.set_yscale("symlog", linthresh=lim/1000)
        ax.axhline(y=0, ls="--", color="k", zorder=-1)
        ax.axvline(x=0, ls="--", color="k", zorder=-1)

        GSP = np.linspace(-lim, +lim)
        ax.plot(GSP, -GSP, ls="--", color="gray", zorder=-1)
    axd[0].legend(loc="upper right", bbox_to_anchor=(-.25, 1), ncol=1)
    if resolution:
        fig.savefig(f"figures_check/shearprods_{resolution}.png")
    else:
        fig.savefig(f"figures_check/shearprods.png")
    pause
    #---
