import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm
from aux02_plotting import BuRd, letterize
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

modifier = ""
slice_name = "xyi"
Ro_h = 0.2

#+++ Read and reindex dataset
snaps = xr.open_dataset(f"data_post/{slice_name}_snaps{modifier}.nc").chunk(time="auto", Fr_h=1, Ro_h=1)
snaps = snaps.reindex(Ro_h = list(reversed(snaps.Ro_h)))
snaps = snaps.sel(xC = slice(-snaps.headland_intrusion_size_max/3, np.inf),
                  yC = slice(-snaps.L, np.inf))

snaps = snaps.isel(time=-1)

try:
    snaps = snaps.reset_coords(("zC", "zF"))
except ValueError:
    pass
#---

#+++ Options
cbar_kwargs = dict(location="right", shrink=0.5, fraction=0.012, pad=0.02, aspect=30)
figsize = (9, 6.5)

#plot_kwargs = dict(vmin=-0.005, vmax=0.005, cmap=plt.cm.RdBu_r, rasterized=True)
plot_kwargs = dict(vmin=-0.003, vmax=0.003, cmap=BuRd, rasterized=True)
#---

#+++ Create ageostrophic variables and pick subset of simulations
snaps["∂Uᵃ∂z"] = snaps["∂u∂z"] - snaps["∂Uᵍ∂z"]

labels = ["$\partial u/\partial z$", f"$\partial U^a/\partial z$"]

snaps = snaps.sel(Ro_h=Ro_h)
snaps.xC.attrs = dict(long_name="$x$", units="m")
snaps.yC.attrs = dict(long_name="$y$", units="m")
#---

#+++ Plotting loop
fig, axes = plt.subplots(ncols=len(snaps.Fr_h), nrows=2, sharex=True, sharey=True, figsize=figsize)
for j_Fr, Fr_h in enumerate(snaps.Fr_h.values):
    print(f"Plotting Frₕ = {Fr_h}")

    ax = axes[0, j_Fr]
    im = snaps["∂u∂z"].sel(Fr_h=Fr_h).pnplot(ax=ax, x="x", add_colorbar=False, **plot_kwargs)
    ax.set_title(f"$Fr_h=$ {Fr_h}")
    ax.set_xlabel("")

    ax = axes[1, j_Fr]
    im = snaps["∂Uᵃ∂z"].sel(Fr_h=Fr_h).pnplot(ax=ax, x="x", add_colorbar=False, **plot_kwargs)
    ax.set_title("")

    if j_Fr>0:
        for ax in axes[:, j_Fr]:
            ax.set_ylabel("")

    if j_Fr == (len(snaps.Fr_h)-1):
        for ax, label in zip(axes[:, j_Fr], labels):
            ax2 = ax.twinx()
            ax2.set_ylabel(label, fontsize=11)
            ax2.tick_params(left=False, right=False, bottom=False, labelleft=False, labelright=False, labelbottom=False)
            ax2.spines['top'].set_visible(False)
#---

#+++ Prettify and save
opts_land = dict(cmap="Set2", vmin=0, vmax=1, alpha=1.0, zorder=10,)
for ax in axes.flatten():
    ax.pcolormesh(snaps.xC, snaps.yC, snaps.land_mask.where(snaps.land_mask==1), rasterized=True, **opts_land)

fig.colorbar(im, ax=axes.ravel().tolist(), label="Vertical shear [1/s]", **cbar_kwargs)
fig.suptitle("")
fig.get_layout_engine().set(w_pad=0.02, h_pad=0, hspace=0, wspace=0)
letterize(axes.flatten(), x=0.05, y=0.9)
fig.savefig(f"figures/ageo_shears_comparison_Roh={snaps.Ro_h.item()}_{slice_name}{modifier}.pdf", dpi=200)
#---
