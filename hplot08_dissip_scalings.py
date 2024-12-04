import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from aux00_utils import simnames, collect_datasets
from aux02_plotting import letterize, create_mc, mscatter

modifier = ""

simnames_filtered = [ f"{simname}{modifier}" for simname in simnames ]
bulk = collect_datasets(simnames_filtered, slice_name="bulkstats")
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h))).mean("yC")
bulk = create_mc(bulk)

#+++ Define new variables
bulk["γᵇ"] = bulk["⟨ε̄ₚ⟩ᵇ"] / (bulk["⟨ε̄ₚ⟩ᵇ"] + bulk["⟨ε̄ₖ⟩ᵇ"])

bulk["H"]  = bulk.α * bulk.L
bulk["ℰₖ"] = bulk["∫∫∫ᵇε̄ₖdxdydz"]     / (bulk["V∞"]**3 * bulk.L * bulk.H)
bulk["𝒟"] = bulk["∫∫∫⁰⟨∂ᵢ(uᵢp)⟩ₜdxdydz_formdrag"] / (bulk["V∞"]**3 * bulk.L * bulk.H)
#---

#+++ Choose buffers and set some attributes
bulk.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
bulk["ℰₖ"].attrs = dict(long_name="Normalized integrated\nKE dissipation rate, $\\mathcal{E}_k$")
bulk["𝒟"].attrs = dict(long_name="Normalized integrated\nform drag work, $\\mathcal{D}$")
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
                             sharex=True, sharey=False,
                             constrained_layout=True)
    axesf = axes.flatten()
    #---

    #+++ Auxiliary continuous variables
    S_Bu = np.logspace(np.log10(bulk_buff["Slope_Bu"].min())+1/3, np.log10(bulk_buff["Slope_Bu"].max())-1/3)
    rates_curve = 0.1*S_Bu
    #---

    #+++ Plot stuff
    print("Plotting axes 0")
    ax = axesf[0]
    xvarname = "Slope_Bu"
    yvarname = "ℰₖ"
    mscatter(x=bulk_buff[xvarname].values.flatten(), y=bulk_buff[yvarname].values.flatten(), color=bulk.color.values.flatten(), markers=bulk.marker.values.flatten(), ax=ax)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, rates_curve, ls="--", label=r"0.1 $S_h$", color="k")

    print("Plotting axes 1")
    ax = axesf[1]
    xvarname = "Slope_Bu"
    yvarname = "𝒟"
    mscatter(x=bulk_buff[xvarname].values.flatten(), y=bulk_buff[yvarname].values.flatten(), color=bulk.color.values.flatten(), markers=bulk.marker.values.flatten(), ax=ax)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(1e-1, 10)
    ax.plot(S_Bu, S_Bu, ls="--", label=r"$S_h$", color="gray")
    #---

    #+++ Prettify and save
    for ax in axesf:
        ax.legend(loc="lower right")
        ax.grid(True)
        ax.set_title("")
        ax.set_xlabel("$S_h$")

    letterize(axesf, x=0.05, y=0.9, fontsize=14)
    fig.savefig(f"figures/dissip_scalings_buffer={buffer}m{modifier}.pdf")
    #---

