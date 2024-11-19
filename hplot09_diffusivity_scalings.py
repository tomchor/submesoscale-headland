import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from aux02_plotting import letterize, create_mc, mscatter

modifier = ""

bulk = xr.open_dataset(f"data_post/bulkstats_snaps{modifier}.nc", chunks={})
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h))).mean("yC")
bulk = create_mc(bulk)

#+++ Define new variables
bulk["γᵇ"] = bulk["⟨ε̄ₚ⟩ᵇ"] / (bulk["⟨ε̄ₚ⟩ᵇ"] + bulk["⟨ε̄ₖ⟩ᵇ"])
bulk["RoFr"] = bulk.Ro_h * bulk.Fr_h

bulk["⟨⟨w′b′⟩ₜ⟩ᵇ + ⟨Π⟩ᵇ"] = bulk["⟨⟨w′b′⟩ₜ⟩ᵇ"] + bulk["⟨Π⟩ᵇ"]

bulk["H"]  = bulk.α * bulk.L
bulk["ℰₚ"] = bulk["∫∫∫ᵇε̄ₚdxdydz"]     / (bulk["V∞"]**3 * bulk.L * bulk.H)
bulk["w'b'"] = bulk["∫∫∫ᵇ⟨w′b′⟩ₜdxdydz"] / (bulk.L**2 * bulk.H)
bulk["𝒦ʷᵇ"] = (-bulk["w'b'"] / bulk["N²∞"]) / (bulk["V∞"] * bulk.L**3 * bulk.H)
bulk["𝒦"] = (bulk["∫∫∫ᵇε̄ₚdxdydz"] / bulk["N²∞"]) / (bulk["V∞"] * bulk.L**3 * bulk.H)
#---

#+++ Choose buffers and set some attributes
bulk.RoFr.attrs = dict(long_name="$Ro_h Fr_h$")
bulk.Slope_Bu.attrs =  dict(long_name=r"$S_h$")
bulk["ℰₚ"].attrs = dict(long_name="Normalized integrated\nbuoyancy mixing rate, $\\mathcal{E}_p$")
bulk["Kb′"].attrs = dict(long_name="$K_b = -\overline{w′b′} / N^2_\infty$ [m²/s]")
bulk["𝒦"].attrs = dict(long_name="Normalized buoyancy diffusivity $\\mathcal{K}_b$")
bulk["⟨⟨w′b′⟩ₜ⟩ᵇ"].attrs = dict(long_name=r"$\langle\overline{w'b'}\rangle$ [m²/s³]")
bulk["⟨ε̄ₚ⟩ᵇ"].attrs = dict(long_name=r"$\langle\overline{\varepsilon}_p\rangle$ [m²/s³]")
#---

for buffer in bulk.buffer.values[1:]:
    print(f"Plotting with buffer = {buffer} m")
    bulk_buff = bulk.sel(buffer=buffer)

    #+++ Create figure
    nrows = 2
    ncols = 1
    size = 3
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize = (2*ncols*size, nrows*size),
                             sharex=False, sharey=False,
                             constrained_layout=True)
    axesf = axes.flatten()
    #---

    #+++ Auxiliary continuous variables
    RoFr = np.logspace(np.log10(bulk_buff.RoFr.min())+1/2, np.log10(bulk_buff.RoFr.max())-1/2)
    S_Bu = np.logspace(np.log10(bulk_buff["Slope_Bu"].min())+1/3, np.log10(bulk_buff["Slope_Bu"].max())-1/3)
    rates_curve = 0.1*S_Bu
    #---

    #+++ Plot stuff
    print("Plotting axes 0")
    ax = axesf[0]
    xvarname = "Slope_Bu"
    yvarname = "ℰₚ"
    mscatter(x=bulk_buff[xvarname].values.flatten(), y=bulk_buff[yvarname].values.flatten(), color=bulk.color.values.flatten(), markers=bulk.marker.values.flatten(), ax=ax)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, rates_curve, ls="--", label=r"0.1 $S_h$", color="k")
    ax.plot(S_Bu, 0.02*S_Bu, ls="--", label=r"0.02 $S_h$", color="gray")
    ax.legend(loc="lower right")


    print("Plotting axes 1")
    ax = axesf[1]
    xvarname = "RoFr"
    yvarname = "𝒦"
    ax.set_title(bulk_buff[yvarname].attrs["long_name"])
    mscatter(x=bulk_buff[xvarname].values.flatten(), y=bulk_buff[yvarname].values.flatten(), color=bulk.color.values.flatten(), markers=bulk.marker.values.flatten(), ax=ax)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(RoFr, 5.e-4*RoFr, ls="--", label=r"$5\times10^{-4}Ro_h Fr_h$", color="gray", zorder=0)
    #ax.plot(RoFr, 1.e-2*RoFr**2, ls="--", label=r"$2.5\times10^{-4}(Ro_h Fr_h)^2$", color="k", zorder=0)
    ax.legend(loc="lower right")
    #---

    #+++ Prettify and save
    for ax in axesf:
        ax.grid(True)
        ax.set_title("")
    
    letterize(axesf, x=0.05, y=0.9, fontsize=14)
    fig.savefig(f"figures/diffusivity_scalings_buffer={buffer}m{modifier}.pdf")
    #---
