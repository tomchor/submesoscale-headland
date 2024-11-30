import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm
from aux00_utils import simnames, collect_datasets
from aux02_plotting import BuRd, letterize
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

modifier = ""
slice_name = "tafields"
Fr_h = 0.08

#+++ Read and reindex dataset
simnames_filtered = [ f"{simname}{modifier}" for simname in simnames ]
snaps = collect_datasets(simnames_filtered, slice_name=slice_name)
snaps = snaps.sel(xC = slice(-snaps.headland_intrusion_size_max/3, np.inf),
                  yC = slice(-snaps.L, np.inf), Ro_h = slice(0.2, None))

try:
    snaps = snaps.reset_coords(("zC", "zF"))
except ValueError:
    pass
#---

#+++ Options
cbar_kwargs = dict(location="right", shrink=0.5, fraction=0.012, pad=0.02, aspect=30)
figsize = (8, 7)

#plot_kwargs = dict(vmin=-0.005, vmax=0.005, cmap=plt.cm.RdBu_r, rasterized=True)
plot_kwargs = dict(vmin=-1.5e-9, vmax=1.5e-9, cmap=cm.balance, rasterized=True)
#---

#+++ Create ageostrophic variables and pick subset of simulations
snaps["Π"] = snaps.SPR.sum("j")
snaps["Πₕ"] = snaps.SPR.sel(j=[1,2]).sum("j")

variables = ["Π", "Πₕ"]
snaps["Π"].attrs = dict(long_name = r"Total shear production rate $\Pi$ [m²/s³]")
snaps["Πₕ"].attrs = dict(long_name = r"Horizontal shear production rate [m²/s³]")

snaps = snaps.sel(Fr_h=Fr_h)
snaps.xC.attrs = dict(long_name="$x$", units="m")
snaps.yC.attrs = dict(long_name="$y$", units="m")
#---

#+++ Plotting loop
fig, axes = plt.subplots(ncols=len(snaps.Ro_h), nrows=2, sharex=True, sharey=True, figsize=figsize)
for j_Ro, Ro_h in enumerate(snaps.Ro_h.values):
    print(f"Plotting Roₕ = {Ro_h}")

    for i, variable in enumerate(variables):
        ax = axes[i, j_Ro]
        ct = snaps["q̄"].sel(Ro_h=Ro_h).pncontour(ax=ax, x="x", add_colorbar=False, levels=[0], zorder=10, linestyles="--", colors="green")
        im = snaps[variable].sel(Ro_h=Ro_h).pnplot(ax=ax, x="x", add_colorbar=False, **plot_kwargs)
        if i==0:
            ax.set_title(f"$Ro_h=$ {Ro_h}, $S_h=$ {Ro_h/Fr_h}")
            ax.set_xlabel("")
        else:
            ax.set_title("")


    if j_Ro>0:
        for ax in axes[:, j_Ro]:
            ax.set_ylabel("")

    if j_Ro == (len(snaps.Ro_h)-1):
        for i, ax in enumerate(axes[:, j_Ro]):
            ax2 = ax.twinx()
            label = snaps[variables[i]].attrs["long_name"]
            ax2.set_ylabel(label, fontsize=11)
            ax2.tick_params(left=False, right=False, bottom=False, labelleft=False, labelright=False, labelbottom=False)
            ax2.spines['top'].set_visible(False)
#---

#+++ Prettify and save
opts_land = dict(cmap="Set2", vmin=0, vmax=1, alpha=1.0, zorder=10,)
for ax in axes.flatten():
    ax.pcolormesh(snaps.xC, snaps.yC, snaps.land_mask.where(snaps.land_mask==1), rasterized=True, **opts_land)

fig.colorbar(im, ax=axes.ravel().tolist(), **cbar_kwargs)
fig.suptitle("")
fig.get_layout_engine().set(w_pad=0.02, h_pad=0, hspace=0, wspace=0)
letterize(axes.flatten(), x=0.05, y=0.9)
fig.savefig(f"figures/SPR_components_comparison_Frₕ={snaps.Fr_h.item()}_{slice_name}{modifier}.pdf", dpi=200)
#---
