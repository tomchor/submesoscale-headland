import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from scipy.optimize import curve_fit
from aux02_plotting import letterize

modifier = ""

tafields = xr.open_dataset(f"data_post/tafields_snaps{modifier}.nc", decode_times=False)
tafields = tafields.reindex(Ro_h = list(reversed(tafields.Ro_h)))

bulk = xr.open_dataset(f"data_post/bulkstats_snaps{modifier}.nc", chunks={})
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h))).mean("yC")

#+++ Define new variables
bulk["γᵇ"] = bulk["⟨ε̄ₚ⟩ᵇ"] / (bulk["⟨ε̄ₚ⟩ᵇ"] + bulk["⟨ε̄ₖ⟩ᵇ"])
bulk["RoFr"] = bulk.Ro_h * bulk.Fr_h
bulk["RoRi"] = bulk.Ro_h / bulk.Fr_h**2

bulk["∫∫∫ᵇΠdxdydz"] = bulk["⟨Π⟩ᵇ"] * bulk["∫∫∫ᵇ1dxdydz"]

bulk["⟨ε̄ₖ⟩ᴮᴸ"] = bulk["⟨ε̄ₖ⟩ᵇ"].sel(buffer=0) - bulk["⟨ε̄ₖ⟩ᵇ"]
bulk["εₖ_ratio_bl_to_rest"] = bulk["⟨ε̄ₖ⟩ᴮᴸ"] / bulk["⟨ε̄ₖ⟩ᵇ"]

bulk["⟨⟨w′b′⟩ₜ⟩ᵇ + ⟨Π⟩ᵇ"] = bulk["⟨⟨w′b′⟩ₜ⟩ᵇ"] + bulk["⟨Π⟩ᵇ"]
bulk["∫∫∫ᵇ⟨w′b′⟩ₜdxdydz + ∫∫∫ᵇΠdxdydz"] = bulk["∫∫∫ᵇ⟨w′b′⟩ₜdxdydz"] + bulk["∫∫∫ᵇΠdxdydz"]
#---

#+++ Choose buffers and set some attributes
bulk.RoFr.attrs = dict(long_name="$Ro_h Fr_h$")
bulk.RoRi.attrs = dict(long_name="$Ro_h / Fr_h^2$")
bulk.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
bulk["∫∫∫ᵇε̄ₖdxdydz"].attrs = dict(long_name=r"$\int\int\int\overline{\varepsilon}_k dV$ [m⁶/s²]")
bulk["∫∫∫ᵇε̄ₚdxdydz"].attrs = dict(long_name=r"$\int\int\int\overline{\varepsilon}_p dV$ [m⁶/s²]")
bulk["∫∫∫ᵇΠdxdydz"].attrs = dict(long_name=r"$\int\int\int\Pi dV$ [m⁶/s²]")
bulk["∫∫∫ᵇ⟨Ek′⟩ₜdxdydz"].attrs = dict(long_name=r"$\int\int\int {\rm TKE} dV$ [m⁶/s²]")
#---

for buffer in bulk.buffer.values:
    print(f"Plotting with buffer = {buffer} m")
    bulk_buff = bulk.sel(buffer=buffer)

    #+++ Create figure
    nrows = 4
    ncols = 1
    size = 3
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize = (2*ncols*size, nrows*size),
                             sharex=True, sharey=False,
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
    RoFr = np.logspace(np.log10(bulk_buff.RoFr.min())+1/2, np.log10(bulk_buff.RoFr.max())-1/2)
    S_Bu = np.logspace(np.log10(bulk_buff["Slope_Bu"].min())+1/3, np.log10(bulk_buff["Slope_Bu"].max())-1/3)
    rates_curve = 7e-4*S_Bu
    #---

    #+++ Plot stuff
    print("Plotting axes 0")
    ax = axesf[0]
    xvarname = "Slope_Bu"
    yvarname = "∫∫∫ᵇε̄ₖdxdydz"
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, rates_curve, ls="--", label=r"$S_h$", color="k")

    print("Plotting axes 1")
    ax = axesf[1]
    xvarname = "Slope_Bu"
    yvarname = "∫∫∫ᵇε̄ₚdxdydz"
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, rates_curve, ls="--", label=r"$S_h$", color="k")

    print("Plotting axes 2")
    ax = axesf[2]
    xvarname = "Slope_Bu"
    yvarname = "∫∫∫ᵇΠdxdydz"
    #yvarname = "∫∫∫ᵇ⟨w′b′⟩ₜdxdydz + ∫∫∫ᵇΠdxdydz"
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, rates_curve, ls="--", label=r"$S_h$", color="k")

    print("Plotting axes 3")
    ax = axesf[3]
    xvarname = "Slope_Bu"
    yvarname = "∫∫∫ᵇ⟨Ek′⟩ₜdxdydz"
    for cond, label, color, marker in zip(conditions, labels, colors, markers):
        ax.scatter(x=bulk_buff.where(cond)[xvarname], y=bulk_buff.where(cond)[yvarname], label=label, color=color, marker=marker)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, 2e2*S_Bu, ls="--", label=r"$S_h$", color="k")
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

