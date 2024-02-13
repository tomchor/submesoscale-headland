import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from scipy.optimize import curve_fit

use_xyz = True
modifier = "-f2"
if use_xyz:
    print("Using bulk stats calculated with xyz")
    bulk = xr.open_dataset(f"data_post/bulkstats_xyz_snaps{modifier}.nc", chunks={})
else:
    print("Using bulk stats calculated with xyi")
    bulk = xr.open_dataset(f"data_post/bulkstats_snaps{modifier}.nc", chunks={})
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h)))
bulk = bulk.sel(buffer=5)

bulk.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
bulk["⟨εₖ⟩ˣᶻ"] = bulk["∫∫ʷεₖdxdz"] / bulk["∫∫ʷdxdz"]

#+++ Create figure
ncols = 1
nrows = 1
size = 4.0
fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                         figsize = (1.8*ncols*size, nrows*size),
                         sharex=False, sharey=False,
                         squeeze = False,
                         constrained_layout=True)
axesf = axes.flatten()
ax = axesf[0]
#---

cmap = plt.cm.copper_r
cmap = plt.cm.coolwarm
for Ro_h in bulk.Ro_h:
    for Fr_h in bulk.Fr_h:
        bulk0 = bulk.sel(Ro_h=Ro_h, Fr_h=Fr_h)
        S_normalized = (np.log10(bulk0.Slope_Bu) - np.log10(bulk.Slope_Bu).min()) / (2*np.log10(bulk.Slope_Bu).max())
        bulk0["⟨εₖ⟩ˣᶻ"].pnplot(ax=ax, x="y", color=cmap(S_normalized))

norm = LogNorm(vmin=bulk.Slope_Bu.min().values, vmax=bulk.Slope_Bu.max().values)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax, label = "Slope Burger number")

#+++ Prettify and save
for ax in axesf:
    #ax.legend()
    #ax.grid(True)
    ax.axvline(x=0, color="k", ls="--", lw=1, zorder=0)
    ax.set_title(f"x,z average of KE dissipation rate excluding {bulk.buffer.item()} m closest to the boundary")

if modifier:
    fig.savefig(f"figures/dissipation_curves{modifier}.pdf")
else:
    fig.savefig(f"figures/dissipation_curves.pdf")
#---
