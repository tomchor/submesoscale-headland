import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm

modifier = ""

tafields = xr.open_dataset(f"data_post/tafields_snaps{modifier}.nc")
tafields = tafields.reindex(Ro_h = list(reversed(tafields.Ro_h)))
tafields = tafields.sel(yC=slice(0, np.inf))

#+++ Energy transfer variables
tafields["q̄_norm"] = tafields["q̄"] / (tafields["N²∞"] * tafields["f₀"])
tafields["ω̄²"]     = -tafields["f₀"]**2 * tafields.q̄_norm

tafields["Slope_Bu"] = tafields.Ro_h / tafields.Fr_h

tasims = tafields[["q̄", "ω̄²", "ε̄ₖ", "Slope_Bu", "average_CSI_mask", "average_stratification_mask"]].stack(simulation=["Ro_h", "Fr_h"])
tawake = tasims.where(tasims.Slope_Bu>1, drop=True)
taterr = tasims.where(tasims.Fr_h==1.25, drop=True)
#---

#+++ Get rid of multi-index
simnames = [ "R"+str(sim.Ro_h.item())+"F"+str(sim.Fr_h.item()) for sim in tawake.simulation ]
Ro_h_values = tawake.Ro_h.values
Fr_h_values = tawake.Fr_h.values

tawake = tawake.drop_vars({'Fr_h', 'simulation', 'Ro_h'}).assign_coords(simulation=simnames)
tawake["Ro_h"] = xr.DataArray(data=Ro_h_values, dims="simulation", coords=dict(simulation=tawake.simulation))
tawake["Fr_h"] = xr.DataArray(data=Fr_h_values, dims="simulation", coords=dict(simulation=tawake.simulation))

CSI = tawake.where(tawake.average_stratification_mask)

simnames = [ "R"+str(sim.Ro_h.item())+"F"+str(sim.Fr_h.item()) for sim in taterr.simulation ]
Ro_h_values = taterr.Ro_h.values
Fr_h_values = taterr.Fr_h.values

taterr = taterr.drop_vars({'Fr_h', 'simulation', 'Ro_h'}).assign_coords(simulation=simnames)
taterr["Ro_h"] = xr.DataArray(data=Ro_h_values, dims="simulation", coords=dict(simulation=taterr.simulation))
taterr["Fr_h"] = xr.DataArray(data=Fr_h_values, dims="simulation", coords=dict(simulation=taterr.simulation))

SHR = taterr.where(taterr.average_stratification_mask)
#---

#+++ Bin results to get a clearer trend
q̄_bins = -np.logspace(-14, np.log10(-CSI["q̄"].min()), 100)[::-1]
q̄_bin_labels = (q̄_bins[1:] + q̄_bins[:-1]) / 2
CSI_q̄_binned = CSI.groupby_bins("q̄", q̄_bins, labels=q̄_bin_labels).mean()

ω̄_bins = np.logspace(-14, np.log10(CSI["ω̄²"].max()), 100)
ω̄_bin_labels = (ω̄_bins[1:] + ω̄_bins[:-1]) / 2
CSI_ω̄_binned = CSI.groupby_bins("ω̄²", ω̄_bins, labels=ω̄_bin_labels).mean()
#---

#+++ Coarsen points for a not-to-cluttered figure
CSI_coarse = CSI.coarsen(xC=2, yC=2, boundary="pad").max()
SHR_coarse = SHR.coarsen(xC=2, yC=2, boundary="pad").max()
#---

#+++ Plot options and labels
CSI["q̄"].attrs = dict(long_name = "Time-avg PV [1/s³]")
CSI["ω̄²"].attrs = dict(long_name = "ω̄²ₘₐₓ = -f₀ q / N²∞ [1/s²]")
CSI["ε̄ₖ"].attrs = dict(long_name = "KE dissipation rate [m²/s³]")

scatter_options = dict(s=2, edgecolor="none", rasterized=True, facecolor="gray", xscale="symlog", yscale="log")
binned_options = dict(s=30, edgecolor="none", rasterized=True, facecolor="black")

Ri_cbar_options = dict(vmin=0, vmax=10, cmap="copper_r")
Ro_cbar_options = dict(vmin=-10, vmax=+10, cmap=cm.balance)
#---

#+++ Figure and options
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8),
                         constrained_layout=True, squeeze=False)
axesf = axes.flatten()
#---

#+++ Plot scatter plot
ax = axesf[0]
CSI_coarse.plot.scatter(ax=ax, x="q̄", y="ε̄ₖ", **scatter_options)
CSI_q̄_binned.plot.scatter(ax=ax, x="q̄", y="ε̄ₖ", **binned_options)
#ax.set_xlim(None, -1e-11)

ax = axesf[1]
CSI_coarse.plot.scatter(ax=ax, x="ω̄²", y="ε̄ₖ", **scatter_options)
CSI_ω̄_binned.plot.scatter(ax=ax, x="ω̄²", y="ε̄ₖ", **binned_options)
#ax.set_xlim(1e-11, None)

ax = axesf[2]
SHR_coarse.plot.scatter(ax=ax, x="q̄", y="ε̄ₖ", **scatter_options)
#CSI_q̄_binned.plot.scatter(ax=ax, x="q̄", y="ε̄ₖ", **binned_options)
#ax.set_xlim(None, -1e-11)

ax = axesf[3]
SHR_coarse.plot.scatter(ax=ax, x="ω̄²", y="ε̄ₖ", **scatter_options)
#CSI_ω̄_binned.plot.scatter(ax=ax, x="ω̄²", y="ε̄ₖ", **binned_options)
#ax.set_xlim(1e-11, None)
#---

#+++ Prettify axes and save
for ax in axesf:
    ax.set_title("")
    ax.grid(True)
    ax.set_ylim(1e-12, None)
    ax.set_xscale("symlog", linthresh=1e-13)

fig.savefig(f"figures/ω_εₖ_correlation{modifier}.pdf", dpi=200)
#---