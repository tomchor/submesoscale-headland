import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm
from aux02_plotting import BuRd
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

modifier = "-f2"
slice_name = "tafields"
λ = 50

#+++ Read and reindex dataset
xyi_snaps = xr.open_dataset(f"data_post/xyi_snaps{modifier}.nc")
snaps = xr.open_dataset(f"data_post/{slice_name}_snaps{modifier}.nc")
snaps["Ri"] = xyi_snaps.Ri
snaps["Ro"] = xyi_snaps.Ro

snaps = snaps.reindex(Ro_h = list(reversed(snaps.Ro_h)))
snaps = snaps.sel(xC = slice(-snaps.headland_intrusion_size_max/3, np.inf),
                  yC = slice(-snaps.runway_length/2, np.inf))

snaps.xC.attrs = dict(units="m")
snaps.yC.attrs = dict(units="m")

#if 1:
#    snaps = snaps.isel(time=-1)
#else:
#    snaps = snaps.chunk(time="auto").sel(time=slice(None, None, 1)).mean("time", keep_attrs=True)
#    snaps = snaps.expand_dims(time = [0]).isel(time=0)

try:
    snaps = snaps.reset_coords(("zC", "zF"))
except ValueError:
    pass
#---

#+++ Options
cbar_kwargs = dict(location="right", shrink=0.8, fraction=0.04, pad=0.02, aspect=30)
figsize = (10, 8)

#plot_kwargs = dict(vmin=-0.005, vmax=0.005, cmap=plt.cm.RdBu_r, rasterized=True)
plot_kwargs = dict(vmin=0, vmax=1, cmap=BuRd, rasterized=True)
#---

#+++ Create new variables and pick subset of simulations
snaps["q̂_norm"] = snaps["q̂"]  / (snaps["N²∞"] * snaps["f₀"])
snaps["q̂_norm"].attrs = dict(long_name=r"Time-averaged normalized filtered Ertel PV")

snaps["q̄_norm"] = snaps["q̄"]  / (snaps["N²∞"] * snaps["f₀"])
snaps["q̄_norm"].attrs = dict(long_name=r"Time-averaged normalized Ertel PV")

snaps["Rᵍ_PVvs"] = (-snaps.Ri**(-1) / (1 + snaps.Ro - 1/snaps.Ri)).mean("time")
snaps["Rᵍ_PVvs2"] = (-snaps.Ri**(-1) / (1 + snaps.Ro - 1/snaps.Ri)).mean("time").where(snaps.average_CSI_mask)

q̄_vs = snaps["q̄ᵢ"].sel(i=[1,2]).sum("i")
snaps["R_PVvs"] = (q̄_vs / snaps["q̄"])
snaps["R_PVvs2"] = (q̄_vs / snaps["q̄"]).where(snaps.average_CSI_mask)

SP_v = snaps.SP.sel(j=3)
snaps["R_SPv"]  = (SP_v / snaps.Π)
snaps["R_SPv2"] = (SP_v / snaps.Π).where(snaps.average_CSI_mask)

snaps = snaps.sel(Ro_h=1.25, λ=λ)

variables = ["Rᵍ_PVvs2", "R_PVvs2", "R_SPv2"]
labels = [r"$\frac{-Ri^{-1}}{1 + Ro - Ri^{-1}}$", r"$\frac{\tilde q^z}{\tilde q}$", r"$\frac{\Pi^z}{\Pi}$"]
#---

#+++ Plotting loop
fig, axes = plt.subplots(ncols=len(snaps.Fr_h), nrows=3, sharex=True, sharey=True)
contour_kwargs = dict(colors="k", linewidths=0.8, linestyles="--", levels=[0])
for j_Fr, Fr_h in enumerate(snaps.Fr_h.values):
    for i_var, variable in enumerate(variables):
        #print()
        #print("Fr", j_Fr, Fr_h)
        #print("var", i_var, variable)
        ax = axes[i_var, j_Fr]
        im = snaps[variable].sel(Fr_h=Fr_h).pnplot(ax=ax, x="x", add_colorbar=False, **plot_kwargs)
        ct = snaps["q̂_norm"].sel(Fr_h=Fr_h).pncontour(ax=ax, x="x", add_colorbar=False, zorder=5, **contour_kwargs)

        if i_var == 0:
            ax.set_title(f"$Fr_h=$ {Fr_h}")
        else:
            ax.set_title("")

        if i_var < len(variables)-1:
            ax.set_xlabel("")

        if j_Fr>0:
            ax.set_ylabel("")
            ax.set_yticklabels([])

        if j_Fr == (len(snaps.Fr_h)-1):
            ax2 = ax.twinx()
            ax2.set_ylabel(labels[i_var], fontsize=11)
            ax2.tick_params(left=False, right=False, bottom=False, labelleft=False, labelright=False, labelbottom=False)
            ax2.spines['top'].set_visible(False)

            fig.colorbar(im, ax=axes[i_var, :].ravel().tolist(), label="Ratios [-]", **cbar_kwargs)
#---

#+++ Prettify and save
opts_land = dict(cmap="Set2", vmin=0, vmax=1, alpha=1.0, zorder=10,)
for ax in axes.flatten():
    ax.pcolormesh(snaps.xC, snaps.yC, snaps.land_mask.where(snaps.land_mask==1), rasterized=True, **opts_land)

fig.suptitle("")
fig.get_layout_engine().set(w_pad=0.02, h_pad=0, hspace=0, wspace=0)
fig.savefig(f"figures/ageo_SP_comparison_λ={λ}_{slice_name}{modifier}.pdf", dpi=200)
#---
