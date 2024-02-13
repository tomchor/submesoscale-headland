import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
π = np.pi

resolution = ""
slice_name = "xyi"
n_cycles = 1/2

snaps = xr.load_dataset(f"data_post/{slice_name}_snaps_{resolution}.nc")[["PV", "Ri", "Ro", "εₖ", "εₚ"]]

t_slice = slice(snaps.time[-1] - n_cycles, np.inf)
snaps = snaps.sel(time=t_slice)

CSI = snaps.where(np.logical_and(snaps.PV < 0, snaps.Ri > 0), drop=True)
CSI["PV_abs"] = abs(CSI.PV)
CSI.PV_abs.attrs = dict(long_name="negative Ertel Potential Vorticity")

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 8),
                         constrained_layout = True, sharex = "col")
scatter_options = dict(s=10, xscale="log", yscale="log", rasterized=True,
                       edgecolor="none")

Ri_cbar_options = dict(vmin=0, vmax=10, cmap="copper_r")
Ro_cbar_options = dict(vmin=-10, vmax=+10, cmap=cm.balance)

#+++ Bin results to get a clearer trend
bins = np.logspace(np.log10(CSI.PV_abs.min()), np.log10(CSI.PV_abs.max()), 100)
bins_labels = (bins[1:] + bins[:-1]) / 2
CSI_binned = CSI.groupby_bins("PV_abs", bins, labels=bins_labels).median()
#---

#+++ Plot scatter plot
CSI.plot.scatter(ax=axes[0,0], x="PV_abs", y="εₖ", hue="Ri", **Ri_cbar_options, **scatter_options)
CSI.plot.scatter(ax=axes[0,1], x="PV_abs", y="εₖ", hue="Ro", **Ro_cbar_options, **scatter_options)
CSI.plot.scatter(ax=axes[1,0], x="PV_abs", y="εₚ", hue="Ri", **Ri_cbar_options, **scatter_options)
CSI.plot.scatter(ax=axes[1,1], x="PV_abs", y="εₚ", hue="Ro", **Ro_cbar_options, **scatter_options)
#---

#+++ Plot binned data
for ax in axes[0,:]:
    CSI_binned.plot.scatter(ax=ax, x="PV_abs", y="εₖ", s=30, c="r")
for ax in axes[1,:]:
    CSI_binned.plot.scatter(ax=ax, x="PV_abs", y="εₚ", s=30, c="r")
#---

#+++ Prettify axes
for ax in axes.flatten():
    ax.set_title("")
    ax.set_xlim(1e-12, None)
    ax.set_ylim(1e-13, None)
    ax.grid(True)
#---

#+++
if resolution:
    fig.savefig(f"figures_check/PV_dissip_{resolution}.png")
else:
    fig.savefig(f"figures_check/PV_dissip.png")
#---
