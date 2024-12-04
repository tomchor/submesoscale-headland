import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from aux02_plotting import letterize, create_mc, mscatter

resolution = ""

bulk_ac = xr.open_dataset(f"data_post/bulkstats_snaps{resolution}.nc", chunks={})
bulk_ac = bulk_ac.reindex(Ro_h = list(reversed(bulk_ac.Ro_h))).mean("yC")
bulk_ac = create_mc(bulk_ac)

bulk_cy = xr.open_dataset(f"data_post/bulkstats_snaps-S{resolution}.nc", chunks={})
bulk_cy = bulk_cy.reindex(Ro_h = list(reversed(bulk_cy.Ro_h))).mean("yC")
bulk_cy = create_mc(bulk_cy)

#+++ Define new variables
bulk_ac["γᵇ"] = bulk_ac["⟨ε̄ₚ⟩ᵇ"] / (bulk_ac["⟨ε̄ₚ⟩ᵇ"] + bulk_ac["⟨ε̄ₖ⟩ᵇ"])

bulk_ac["∫∫∫ᵇΠdxdydz"] = bulk_ac["⟨Π⟩ᵇ"] * bulk_ac["∫∫∫ᵇ1dxdydz"]

bulk_ac["⟨ε̄ₖ⟩ᴮᴸ"] = bulk_ac["⟨ε̄ₖ⟩ᵇ"].sel(buffer=0) - bulk_ac["⟨ε̄ₖ⟩ᵇ"]
bulk_ac["εₖ_ratio_bl_to_rest"] = bulk_ac["⟨ε̄ₖ⟩ᴮᴸ"] / bulk_ac["⟨ε̄ₖ⟩ᵇ"]

bulk_ac["H"]  = bulk_ac.α * bulk_ac.L
bulk_ac["ℰₖ"] = bulk_ac["∫∫∫ᵇε̄ₖdxdydz"]     / (bulk_ac["V∞"]**3 * bulk_ac.L * bulk_ac.H)
bulk_ac["ℰₚ"] = bulk_ac["∫∫∫ᵇε̄ₚdxdydz"]     / (bulk_ac["V∞"]**3 * bulk_ac.L * bulk_ac.H)

bulk_cy["H"]  = bulk_cy.α * bulk_cy.L
bulk_cy["ℰₖ"] = bulk_cy["∫∫∫ᵇε̄ₖdxdydz"]     / (bulk_cy["V∞"]**3 * bulk_cy.L * bulk_cy.H)
bulk_cy["ℰₚ"] = bulk_cy["∫∫∫ᵇε̄ₚdxdydz"]     / (bulk_cy["V∞"]**3 * bulk_cy.L * bulk_cy.H)
#---

#+++ Choose buffers and set some attributes
bulk_ac.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
bulk_ac["ℰₖ"].attrs = dict(long_name="Normalized integrated\nKE dissipation rate, $\mathcal{E}_k$")
bulk_ac["ℰₚ"].attrs = dict(long_name="Normalized integrated\nbuoyancy mixing rate, $\mathcal{E}_p$")
#---

for buffer in [5]:
    print(f"Plotting with buffer = {buffer} m")
    bulk_buff_ac = bulk_ac.sel(buffer=buffer)
    bulk_buff_cy = bulk_cy.sel(buffer=buffer)

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
    S_Bu = np.logspace(np.log10(bulk_buff_ac["Slope_Bu"].min())+1/3, np.log10(bulk_buff_ac["Slope_Bu"].max())-1/3)
    rates_curve = 0.1*S_Bu
    #---

    #+++ Plot stuff
    print("Plotting axes 0")
    ax = axesf[0]
    xvarname = "Slope_Bu"
    yvarname = "ℰₖ"
    ax.scatter(x=bulk_buff_ac[xvarname].values.flatten(), y=bulk_buff_ac[yvarname].values.flatten(), color="blue", marker="D", alpha=0.6, label="Anticycl. conf.")
    ax.scatter(x=bulk_buff_cy[xvarname].values.flatten(), y=bulk_buff_cy[yvarname].values.flatten(), color="red", marker="X", alpha=0.6, label="Cyclonic conf.")
    ax.set_ylabel(bulk_buff_ac[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff_ac[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, rates_curve, ls="--", label=r"0.1 $S_h$", color="blue")
    ax.plot(S_Bu, 0.02*S_Bu**(1/2), ls="--", label=r"0.02 $S_h^{1/2}$", color="red")

    print("Plotting axes 1")
    ax = axesf[1]
    xvarname = "Slope_Bu"
    yvarname = "ℰₚ"
    ax.scatter(x=bulk_buff_ac[xvarname].values.flatten(), y=bulk_buff_ac[yvarname].values.flatten(), color="blue", marker="D", alpha=0.6, label="Anticyclonic conf.")
    ax.scatter(x=bulk_buff_cy[xvarname].values.flatten(), y=bulk_buff_cy[yvarname].values.flatten(), color="red", marker="X", alpha=0.6, label="Cyclonic conf.")
    ax.set_ylabel(bulk_buff_ac[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_buff_ac[xvarname].attrs["long_name"])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.plot(S_Bu, 0.02*S_Bu, ls="--", label=r"0.02 $S_h$", color="blue")
    ax.plot(S_Bu, 0.007*S_Bu**(1/2), ls="--", label=r"0.007 $S_h^{1/2}$", color="red")
    #---

    #+++ Prettify and save
    for ax in axesf:
        ax.legend(loc="lower right")
        ax.grid(True)
        ax.set_title("")
        ax.set_xlabel("$S_h$")
    
    letterize(axesf, x=0.05, y=0.9, fontsize=14)
    fig.savefig(f"figures/dissip_cyclonic_buffer={buffer}m{resolution}.pdf")
    #---

