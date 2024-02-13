import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm
from aux02_plotting import BuRd
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

modifier = ""
slice_name = "xyi"

#+++ Read and reindex dataset
snaps = xr.open_dataset(f"data_post/{slice_name}_snaps{modifier}.nc")
snaps = snaps.reindex(Ro_h = list(reversed(snaps.Ro_h)))
snaps = snaps.sel(xC = slice(-snaps.headland_intrusion_size_max/3, np.inf),
                  yC = slice(-snaps.runway_length/2, np.inf))

if 1:
    snaps = snaps.isel(time=-1)
else:
    snaps = snaps.chunk(time="auto").sel(time=slice(None, None, 1)).mean("time", keep_attrs=True)
    snaps = snaps.expand_dims(time = [0]).isel(time=0)

try:
    snaps = snaps.reset_coords(("zC", "zF"))
except ValueError:
    pass
#---

#+++ Options
cbar_kwargs = dict(location="right", shrink=0.5, fraction=0.012, pad=0.02, aspect=30)
figsize = (10, 8)

#plot_kwargs = dict(vmin=-0.005, vmax=0.005, cmap=plt.cm.RdBu_r, rasterized=True)
plot_kwargs = dict(vmin=-0.003, vmax=0.003, cmap=BuRd, rasterized=True)
#---

#+++ Create ageostrophic variables and pick subset of simulations
snaps["Uᵃ"] = snaps.u - snaps["Uᵍ"]
snaps["∂Uᵃ∂z"] = snaps["∂u∂z"] - snaps["∂Uᵍ∂z"]

snaps = snaps.sel(Ro_h=0.5)
#---

#+++ Plotting loop
fig, axes = plt.subplots(ncols=len(snaps.Fr_h), nrows=2, sharex=True, sharey=True)
for j_Fr, Fr_h in enumerate(snaps.Fr_h.values):
    ax0 = axes[0, j_Fr]
    im = snaps["∂u∂z"].sel(Fr_h=Fr_h).pnplot(ax=ax0, x="x", add_colorbar=False, **plot_kwargs)
    ax0.set_title(f"$Fr_h=$ {Fr_h}")
    ax0.set_xlabel("")
    ax0.set_xticklabels([])


    ax1 = axes[1, j_Fr]
    im = snaps["∂Uᵃ∂z"].sel(Fr_h=Fr_h).pnplot(ax=ax1, x="x", add_colorbar=False, **plot_kwargs)
    ax1.set_title("")

    if j_Fr>0:
        for ax in axes[:, j_Fr]:
            ax.set_ylabel("")
            ax.set_yticklabels([])

    if j_Fr == (len(snaps.Fr_h)-1):
        for ax, label in zip([ax0, ax1], ["$\partial u/\partial z$", f"$\partial U^a/\partial z$"]):
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
fig.savefig(f"figures/ageo_shears_comparison_{slice_name}{modifier}.pdf", dpi=200)
#---
