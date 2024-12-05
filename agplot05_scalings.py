import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from aux00_utils import simnames, collect_datasets
from aux02_plotting import letterize, create_mc, mscatter


#+++ Open datasets
modifier = ""
bulk_ac = xr.open_dataset(f"data_post/bulkstats_snaps{modifier}.nc", chunks={})
bulk_ac = bulk_ac.reindex(Ro_h = list(reversed(bulk_ac.Ro_h))).mean("yC")
bulk_ac = create_mc(bulk_ac)

modifier = "-S"
simnames_filtered = [ f"{simname}{modifier}" for simname in simnames ]
bulk_cy = collect_datasets(simnames_filtered, slice_name="bulkstats")
bulk_cy = bulk_cy.reindex(Ro_h = list(reversed(bulk_cy.Ro_h))).mean("yC")
bulk_cy = create_mc(bulk_cy)
#---

#+++ Define new variables
bulk_ac["H"]  = bulk_ac.α * bulk_ac.L
bulk_ac["ℰₖ"] = bulk_ac["∫∫∫ᵇε̄ₖdxdydz"]     / (bulk_ac["V∞"]**3 * bulk_ac.L * bulk_ac.H)
bulk_ac["ℰₚ"] = bulk_ac["∫∫∫ᵇε̄ₚdxdydz"]     / (bulk_ac["V∞"]**3 * bulk_ac.L * bulk_ac.H)

bulk_cy["H"]  = bulk_cy.α * bulk_cy.L
bulk_cy["ℰₖ"] = bulk_cy["∫∫∫ᵇε̄ₖdxdydz"]     / (bulk_cy["V∞"]**3 * bulk_cy.L * bulk_cy.H)
bulk_cy["ℰₚ"] = bulk_cy["∫∫∫ᵇε̄ₚdxdydz"]     / (bulk_cy["V∞"]**3 * bulk_cy.L * bulk_cy.H)
#---

#+++ Set some attributes
bulk_ac.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
bulk_ac["ℰₖ"].attrs = dict(long_name="Normalized integrated\nKE dissipation rate, $\\mathcal{E}_k$")
bulk_ac["ℰₚ"].attrs = dict(long_name="Normalized integrated\nbuoyancy mixing rate, $\\mathcal{E}_p$")

bulk_cy.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
bulk_cy["ℰₖ"].attrs = dict(long_name="Normalized integrated\nKE dissipation rate, $\mathcal{E}_k$")
bulk_cy["ℰₚ"].attrs = dict(long_name="Normalized integrated\nbuoyancy mixing rate, $\mathcal{E}_p$")
#---

bulk_ac = bulk_ac.sel(buffer=5)
bulk_cy = bulk_ce.sel(buffer=5)

#+++ Create figure
nrows = 1
ncols = 2
size = 3
fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                         figsize = (1.7*ncols*size, nrows*size),
                         sharex=True, sharey=True,
                         constrained_layout=True)
axesf = axes.flatten()
#---

#+++ Auxiliary continuous variables
S_Bu = np.logspace(np.log10(bulk_ac["Slope_Bu"].min())+1/3, np.log10(bulk_ac["Slope_Bu"].max())-1/3)
rates_curve = 0.1*S_Bu
#---

#+++ Plot stuff
print("Plotting axes 0")
ax = axesf[0]
xvarname = "Slope_Bu"
yvarname = "ℰₖ"
mscatter(x=bulk_ac[xvarname].values.flatten(), y=bulk_ac[yvarname].values.flatten(), color=bulk_ac.color.values.flatten(), markers=bulk_ac.marker.values.flatten(), ax=ax)
ax.set_ylabel(bulk_ac[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_ac[xvarname].attrs["long_name"])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(2e-4, 1)
ax.plot(S_Bu, rates_curve, ls="--", label=r"0.1 $S_h$", color="k")

print("Plotting axes 1")
ax = axesf[1]
xvarname = "Slope_Bu"
yvarname = "ℰₚ"
mscatter(x=bulk_ac[xvarname].values.flatten(), y=bulk_ac[yvarname].values.flatten(), color=bulk_ac.color.values.flatten(), markers=bulk_ac.marker.values.flatten(), ax=ax)
ax.set_ylabel(bulk_ac[yvarname].attrs["long_name"]); ax.set_xlabel(bulk_ac[xvarname].attrs["long_name"])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylim(2e-4, 1)
ax.plot(S_Bu, 0.02*S_Bu, ls="--", label=r"0.02 $S_h$", color="k")
#---

#+++ Prettify and save
for ax in axesf:
    ax.legend(loc="lower right")
    ax.grid(True)
    ax.set_title("")
    ax.set_xlabel("$S_h$")

fig.savefig(f"figures/poster_scalings_buffer=5m{modifier}.pdf")
#---
