import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from scipy.optimize import curve_fit

modifier = "-f2"

tafields = xr.open_dataset(f"data_post/tafields_snaps{modifier}.nc", decode_times=False)

bulk = xr.open_dataset(f"data_post/bulkstats_snaps{modifier}.nc", chunks={})
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h))).mean("yC")

#+++ Define new variables
bulk["γᵇ"] = bulk["⟨ε̄ₚ⟩ᵇ"] / (bulk["⟨ε̄ₚ⟩ᵇ"] + bulk["⟨ε̄ₖ⟩ᵇ"])

bulk["∫∫∫ᵇΠdxdydz"] = bulk["⟨Π⟩ᵇ"] * bulk["∫∫∫ᵇ1dxdydz"]
#---

#+++ Choose buffers and set some attributes
bulk.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
#---

for buffer in bulk.buffer.values:
    print(f"Plotting with buffer = {buffer} m")
    bulk_buff = bulk.sel(buffer=buffer)

    #+++ Create figure
    ncols = 2
    nrows = 1
    size = 3.
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize = (1.4*ncols*size, nrows*size),
                             sharex=False, sharey=False,
                             constrained_layout=True)
    axesf = axes.flatten()
    #---

    #+++ Marker details
    marker_large_Bu = "^"
    marker_unity_Bu = "s"
    marker_small_Bu = "o"
    markers = [marker_large_Bu, marker_unity_Bu, marker_small_Bu]

    color_large_Bu = "blue"
    color_unity_Bu = "orange"
    color_small_Bu = "green"
    colors = [color_large_Bu, color_unity_Bu, color_small_Bu]
    
    conditions = [bulk_buff.Bu_h>1, bulk_buff.Bu_h==1, bulk_buff.Bu_h<1]
    labels = ["Bu>1", "Bu=1", "Bu<1"]
    #---

    #+++ Auxiliary continuous variables
    S_Bu = np.logspace(np.log10(bulk_buff["Slope_Bu"].min())+1/3, np.log10(bulk_buff["Slope_Bu"].max())-1/3)
    S_Bu_label = r"Slope Burger number $S_h = Ro_h/Fr_h$"
    #---

    #+++ Plot stuff
    print("Plotting axes 0")
    ax = axesf[0]
    xvarname = "Slope_Bu"
    yvarname = "∫∫∫ᵇε̄ₖdxdydz"
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel("Integrated KE dissipation rate"); ax.set_xlabel(S_Bu_label)
    ax.set_xscale("log"); ax.set_yscale("log");
    ax.plot(S_Bu, 7e-4*S_Bu, ls="--", label=r"$S_h^1$", color="k")

    print("Plotting axes 1")
    ax = axesf[1]
    xvarname = "Slope_Bu"
    yvarname = "∫∫∫ᵇε̄ₚdxdydz"
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel("Integrated buoyancy mixing rate"); ax.set_xlabel(S_Bu_label)
    ax.set_xscale("log"); ax.set_yscale("log");
    ax.plot(S_Bu, 1e-4*S_Bu, ls="--", label=r"$S_h^1$", color="k")

    
    #+++ Prettify and save
    fig.suptitle(f"Integrated quantities excluding {buffer} m closest to boundary")
    for ax in axesf:
        ax.legend(loc="lower right")
        ax.set_axisbelow(True)
        ax.grid(True)
        ax.set_title("")
    
    fig.savefig(f"figures/scalings_buffer={buffer}m.pdf")
    #---

