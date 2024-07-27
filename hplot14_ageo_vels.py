import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm
from aux02_plotting import BuRd, letterize
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

modifier = ""
slice_name = "tafields"
Fr_h = 0.08

#+++ Read and reindex dataset
snaps = xr.open_dataset(f"data_post/{slice_name}_snaps{modifier}.nc").chunk(Fr_h=1, Ro_h=1)
snaps = snaps.sel(xC = slice(-snaps.headland_intrusion_size_max/3, np.inf),
                  yC = slice(-snaps.L, np.inf), Ro_h = slice(0.2, None))

if slice_name == "xyi":
    snaps = snaps.sel(time=30, method="nearest")

try:
    snaps = snaps.reset_coords(("zC", "zF"))
except ValueError:
    pass
#---

#+++ Options
cbar_kwargs = dict(location="right", shrink=0.5, fraction=0.012, pad=0.02, aspect=30)
figsize = (8, 7)

plot_kwargs = dict(vmin=-3, vmax=+3, cmap=BuRd, rasterized=True)
#---

#+++ Create ageostrophic variables and pick subset of simulations
if slice_name == "xyi":
    norm = snaps["∂u∂z"].std(("xC", "yC"))
    snaps["∂u∂z_norm"] = snaps["∂u∂z"] / norm # Normalize
    snaps["∂Uᵍ∂z_norm"] = snaps["∂Uᵍ∂z"] / norm # Normalize
else:
    component = dict(i=1, j=3)
    norm = snaps["∂ⱼūᵢ"].sel(**component).std(("xC", "yC"))
    snaps["∂u∂z_norm"] = snaps["∂ⱼūᵢ"].sel(**component) / norm # Normalize
    snaps["∂Uᵍ∂z_norm"] = snaps["∂ⱼŪᵍᵢ"].sel(**component) / norm # Normalize

snaps["∂Uᵃ∂z_norm"] = snaps["∂u∂z_norm"] - snaps["∂Uᵍ∂z_norm"]

labels = ["$\partial \overline{u}/\partial z$", "$\partial \overline{u}^a/\partial z$"]

snaps = snaps.sel(Fr_h=Fr_h)
snaps.xC.attrs = dict(long_name="$x$", units="m")
snaps.yC.attrs = dict(long_name="$y$", units="m")
#---

#+++ Plotting loop
fig, axes = plt.subplots(ncols=len(snaps.Ro_h), nrows=2, sharex=True, sharey=True, figsize=figsize)
for j_Ro, Ro_h in enumerate(snaps.Ro_h.values):
    print(f"Plotting Roₕ = {Ro_h}")

    ax = axes[0, j_Ro]
    im = snaps["∂u∂z_norm"].sel(Ro_h=Ro_h).pnplot(ax=ax, x="x", add_colorbar=False, **plot_kwargs)
    ct = snaps["q̄"].sel(Ro_h=Ro_h).pncontour(ax=ax, x="x", add_colorbar=False, levels=[0], zorder=10, linestyles="--", colors="green")
    ax.set_title(f"$Ro_h=$ {Ro_h}, $S_h=$ {Ro_h/Fr_h}")
    ax.set_xlabel("")

    ax = axes[1, j_Ro]
    im = snaps["∂Uᵃ∂z_norm"].sel(Ro_h=Ro_h).pnplot(ax=ax, x="x", add_colorbar=False, **plot_kwargs)
    ct = snaps["q̄"].sel(Ro_h=Ro_h).pncontour(ax=ax, x="x", add_colorbar=False, levels=[0], zorder=10, linestyles="--", colors="green")
    ax.set_title("")

    if j_Ro>0:
        for ax in axes[:, j_Ro]:
            ax.set_ylabel("")

    if j_Ro == (len(snaps.Ro_h)-1):
        for ax, label in zip(axes[:, j_Ro], labels):
            ax2 = ax.twinx()
            ax2.set_ylabel(label, fontsize=11)
            ax2.tick_params(left=False, right=False, bottom=False, labelleft=False, labelright=False, labelbottom=False)
            ax2.spines['top'].set_visible(False)
#---

#+++ Prettify and save
opts_land = dict(cmap="Set2", vmin=0, vmax=1, alpha=1.0, zorder=10,)
for ax in axes.flatten():
    ax.pcolormesh(snaps.xC, snaps.yC, snaps.land_mask.where(snaps.land_mask==1), rasterized=True, **opts_land)

fig.colorbar(im, ax=axes.ravel().tolist(), label="Normalized vertical shear [-]", **cbar_kwargs)
fig.suptitle("")
fig.get_layout_engine().set(w_pad=0.02, h_pad=0, hspace=0, wspace=0)
letterize(axes.flatten(), x=0.05, y=0.9)
fig.savefig(f"figures/ageo_shears_comparison_Frₕ={snaps.Fr_h.item()}_{slice_name}{modifier}.pdf", dpi=200)
#---
