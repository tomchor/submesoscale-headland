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

bulk["∫∫∫ᵇΠdxdydz"] = bulk["⟨Π⟩ᵇ"] * bulk["∫∫∫ᵇ1dxdydz"]

bulk["⟨ε̄ₖ⟩ᴮᴸ"] = bulk["⟨ε̄ₖ⟩ᵇ"].sel(buffer=0) - bulk["⟨ε̄ₖ⟩ᵇ"]
bulk["εₖ_ratio_bl_to_rest"] = bulk["⟨ε̄ₖ⟩ᴮᴸ"] / bulk["⟨ε̄ₖ⟩ᵇ"]

bulk["H"]  = bulk.α * bulk.L
bulk["ℰₖ"] = bulk["∫∫∫ᵇε̄ₖdxdydz"]     / (bulk["V∞"]**3 * bulk.L * bulk.H)
bulk["ℰₚ"] = bulk["∫∫∫ᵇε̄ₚdxdydz"]     / (bulk["V∞"]**3 * bulk.L * bulk.H)
bulk["𝓅"]  = bulk["∫∫∫ᵇ⟨uᵢ∂ᵢp⟩ₜdxdydz"] / (bulk["V∞"]**3 * bulk.L * bulk.H)
bulk["𝓅2"] = bulk["∫∫∫⁰⟨∂ᵢ(uᵢp)⟩ₜdxdydz_diverg"] / (bulk["V∞"]**3 * bulk.L * bulk.H)
bulk["𝓅3"] = bulk["∫∫∫⁰⟨∂ᵢ(uᵢp)⟩ₜdxdydz_formdrag"] / (bulk["V∞"]**3 * bulk.L * bulk.H)
bulk["𝒜"]  = bulk["∫∫∫ᵇ⟨uᵢ∂ⱼuⱼuᵢ⟩ₜdxdydz"] / (bulk["V∞"]**3 * bulk.L * bulk.H)
bulk["𝒜2"]  = bulk["∫∫∫⁰⟨uᵢ∂ⱼuⱼuᵢ⟩ₜdxdydz_diverg"] / (bulk["V∞"]**3 * bulk.L * bulk.H)
bulk["𝒫"]  = bulk["∫∫∫ᵇΠdxdydz"]      / (bulk["V∞"]**3 * bulk.L * bulk.H)
bulk["𝒦"]  = bulk["∫∫∫ᵇ⟨Ek′⟩ₜdxdydz"] / (bulk["V∞"]**2 * bulk.L**2 * bulk.H)
#---

#+++ Choose buffers and set some attributes
bulk.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
bulk["ℰₖ"].attrs = dict(long_name="Normalized integrated\nKE dissipation rate, $\mathcal{E}_k$")
bulk["ℰₚ"].attrs = dict(long_name="Normalized integrated\nbuoyancy mixing rate, $\mathcal{E}_p$")
bulk["𝓅"].attrs = dict(long_name="Normalized integrated\npressure transport contribution, $\mathcal{p}$")
bulk["𝓅2"].attrs = dict(long_name="Normalized integrated\npressure (divergence), $\mathcal{p}$2")
bulk["𝓅3"].attrs = dict(long_name="Normalized integrated\npressure (form drag), $\mathcal{p}$3")
bulk["𝒜"].attrs = dict(long_name="Normalized integrated\nadvection contribution, $\mathcal{A}$")
bulk["𝒜2"].attrs = dict(long_name="Normalized integrated\nadvection (divergence), $\mathcal{A}$2")
bulk["𝒫"].attrs = dict(long_name="Normalized integrated\nshear production rate, $\mathcal{P}$")
bulk["𝒦"].attrs = dict(long_name="Normalized integrateed\nTKE, $\mathcal{K}$")
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
    yvarname = "ℰₚ"
    mscatter(x=bulk_buff[xvarname].values.flatten(), y=bulk_buff[yvarname].values.flatten(), color=bulk.color.values.flatten(), markers=bulk.marker.values.flatten(), ax=ax)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, rates_curve, ls="--", label=r"0.1 $S_h$", color="k")
    ax.plot(S_Bu, 0.02*S_Bu, ls="--", label=r"0.02 $S_h$", color="gray")

    print("Plotting axes 2")
    ax = axesf[2]
    xvarname = "Slope_Bu"
    yvarname = "𝒫"
    mscatter(x=bulk_buff[xvarname].values.flatten(), y=bulk_buff[yvarname].values.flatten(), color=bulk.color.values.flatten(), markers=bulk.marker.values.flatten(), ax=ax)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, rates_curve, ls="--", label=r"0.1 $S_h$", color="k")
    ax.plot(S_Bu, 0.5*S_Bu, ls="--", label=r"0.5 $S_h$", color="gray")

    print("Plotting axes 3")
    ax = axesf[3]
    xvarname = "Slope_Bu"
    yvarname = "𝒦"
    mscatter(x=bulk_buff[xvarname].values.flatten(), y=bulk_buff[yvarname].values.flatten(), color=bulk.color.values.flatten(), markers=bulk.marker.values.flatten(), ax=ax)
    ax.set_ylabel(bulk_buff[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, 1e0*S_Bu, ls="--", label=r"1 $S_h$", color="gray")
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

