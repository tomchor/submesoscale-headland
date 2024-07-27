import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from aux02_plotting import manual_facetgrid, get_orientation
from cmocean import cm
#plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

modifier = ""

#+++ Read and reindex dataset
snaps = xr.open_dataset(f"data_post/etfields_snaps{modifier}.nc")
snaps = snaps.reindex(Ro_h = list(reversed(snaps.Ro_h)))
snaps = snaps.reindex(Ro_h = list(reversed(snaps.Ro_h)))
snaps = snaps.isel(time=-1).sel(Ro_h=1.25, Fr_h=1.25, λ=[30, 50, 100, 200])

try:
    snaps = snaps.reset_coords(("zC", "zF"))
except ValueError:
    pass
#---

#+++ Adjust/create variables
snaps.PV_norm.attrs = dict(long_name=r"Normalized Ertel PV")

if "q̃" in snaps.variables.keys():
    q̃ = snaps["q̃"].reindex(λ=np.append(0, snaps.λ))
    q̃[dict(λ=0)] = snaps.PV.transpose(*q̃.sel(λ=0).dims)
    q̃_norm = q̃  / (snaps["N²∞"] * snaps["f₀"])
    q̃_norm.attrs = dict(long_name=r"Normalized filtered Ertel PV")

if "wb" in snaps.variables.keys():
    snaps["Kb"] = -snaps.wb / snaps["N²∞"]
#---


fg = q̃_norm.pnplot(x="x", col="λ",
                   cbar_kwargs = dict(shrink=0.6, fraction=0.08, aspect=50, location="right",),
                   #cbar_kwargs = dict(shrink=0.9, fraction=0.03, pad=0.03, aspect=30, location="right",),
                   vmin=-5, vmax=5, cmap="RdBu_r",
                   figsize = (8, 4),
                   rasterized = True)

snaps["land_mask"] = snaps.land_mask.where(snaps.land_mask)
opts_land = dict(cmap="Set2", vmin=0, vmax=1, alpha=1.0, zorder=10)
for i, ax in enumerate(fg.axs.flat):
    if ax.get_xlabel():
        ax.set_xlabel(ax.get_xlabel() + " [m]")
    if ax.get_ylabel():
        ax.set_ylabel(ax.get_ylabel() + " [m]")
    if ax.get_title():
        ax.set_title(ax.get_title() + " m", fontsize=9)

    ax.pcolormesh(snaps.xC, snaps.yC, snaps.land_mask, rasterized=True, **opts_land)
    ax.grid(True)

#+++ Final touches and save
fig = plt.gcf()
#fig.get_layout_engine().set(w_pad=0.02, h_pad=0, hspace=0, wspace=0)
fig.savefig(f"figures/filtered_q_scales{modifier}.pdf", dpi=200)
#---
