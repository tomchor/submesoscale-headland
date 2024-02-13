import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
π = np.pi

#+++ Load condensed dataset
resolution = "-f2"
slice_name = "xyi"
n_cycles = 5

tafields = xr.open_dataset(f"data_post/tafields_snaps{resolution}.nc")
snaps = xr.open_dataset(f"data_post/{slice_name}_snaps{resolution}.nc")#[["Re_b", "PV", "Ri", "Ro", "εₖ", "εₚ", "dbdz", "N²∞",]]
bulk = xr.open_dataset(f"data_post/bulkstats_snaps{resolution}.nc")
#---

#+++ Create new variables and throw out non-CSI simulations
tafields["Slope_Bu"] = tafields.Ro_h / tafields.Fr_h
tafields["γ̄"] = tafields["ε̄ₚ"] / (tafields["ε̄ₖ"] + tafields["ε̄ₚ"])
tafields["Γ̄"] = tafields["ε̄ₚ"] / tafields["ε̄ₖ"]
tafields["F̄rₜ"] = tafields["ε̄ₖ"] / (tafields["Ek′"] * np.sqrt(tafields["N²∞"]))
tafields["RoₕRiₕ_neg"] = -tafields.Ro_h / tafields.Fr_h**2
tafields["PV_h"] = 1 - tafields.Ro_h  - tafields.Fr_h**2
tafields["SP_ratio"] = tafields["SP"].sel(j=[1,2]).sum("j") / tafields["SP"].sel(j=3)

tasims = tafields[["R̄e_b", "F̄rₜ", "Γ̄", "N²∞", "Slope_Bu", "SP_ratio",
                   "RoₕRiₕ_neg", "PV_h", "Ro_h", "Fr_h", "average_CSI_mask"]].stack(simulation=["Ro_h", "Fr_h"])
tawake = tasims.where(tasims.Slope_Bu>=1, drop=True)
#---

#+++ Get rid of multi-index
simnames = [ "R"+str(sim.Ro_h.item())+"F"+str(sim.Fr_h.item()) for sim in tawake.simulation ]

Ro_h_values = tawake.Ro_h.values
Fr_h_values = tawake.Fr_h.values
tawake = tawake.drop_vars({'Fr_h', 'simulation', 'Ro_h'}).assign_coords(simulation=simnames)
tawake["Ro_h"] = xr.DataArray(data=Ro_h_values, dims="simulation", coords=dict(simulation=tawake.simulation))
tawake["Fr_h"] = xr.DataArray(data=Fr_h_values, dims="simulation", coords=dict(simulation=tawake.simulation))
#---

#+++ Pretty names
tawake.R̄e_b.attrs = dict(long_name=r"$Re_b^\text{sgs}$")
tawake.Fr_h.attrs = dict(long_name=r"$Fr_h$")
tawake.Ro_h.attrs = dict(long_name=r"$Ro_h$")
tawake.Γ̄.attrs = dict(long_name=r"Mixing coefficient")
tawake.SP_ratio.attrs = dict(long_name="SPₕₒᵣ / SPᵥₑᵣₜ")
#---

#+++ Narrow down to CSI regions
CSI = tawake.where(tawake.average_CSI_mask)
#---

#+++ Bin dataset
bins_Reb = np.logspace(np.log10(tawake.R̄e_b.where(tawake.R̄e_b > 0).min()), np.log10(tawake.R̄e_b.max()), 100)
bins_Reb_labels = (bins_Reb[1:] + bins_Reb[:-1]) / 2
taw_Reb_binned = tawake[["Γ̄", "R̄e_b"]].groupby_bins("R̄e_b", bins_Reb, labels=bins_Reb_labels).mean()


bins_Frt = np.logspace(np.log10(tawake["F̄rₜ"].where(tawake["F̄rₜ"] > 0).min()), np.log10(tawake["F̄rₜ"].where(np.isfinite(tawake["F̄rₜ"])).max()), 100)
bins_Frt_labels = (bins_Frt[1:] + bins_Frt[:-1]) / 2
taw_Frt_binned = tawake[["Γ̄", "F̄rₜ"]].groupby_bins("F̄rₜ", bins_Frt, labels=bins_Frt_labels).mean()


bins_SPr = np.logspace(np.log10(CSI.SP_ratio.where(CSI.SP_ratio > 0).min()), np.log10(CSI.SP_ratio.max()), 100)
bins_SPr_labels = (bins_SPr[1:] + bins_SPr[:-1]) / 2
CSI_SPr_binned = CSI[["Γ̄", "SP_ratio"]].groupby_bins("SP_ratio", bins_SPr, labels=bins_SPr_labels).mean()
#---

#+++ Create figure and options
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 8),
                         constrained_layout = True, sharey = False)
scatter_options = dict(s=10, xscale="log", rasterized=True,
                       edgecolor="none")

Ri_cbar_options = dict(vmin=0, vmax=2, cmap="copper_r")
Ro_cbar_options = dict(vmin=-10, vmax=+10, cmap=cm.balance)
PV_abs_norm_cbar_options = dict(norm=LogNorm(vmin=1e-8, vmax=1e-4), cmap="copper_r")

Fr_h_cbar_options = dict(norm=LogNorm(vmin=1e-2, vmax=1))
Ro_h_cbar_options = dict(norm=LogNorm(vmin=1e-2, vmax=1))
PV_h_cbar_options = dict(vmin=-1, vmax=+1, cmap=cm.balance)
sim_cbar_options = dict(cmap="tab10")
#---

#+++ Plot the datasets
taw_coarse = tawake.coarsen(xC=20, yC=50, boundary="pad").mean()
CSI_coarse = CSI.coarsen(xC=20, yC=50, boundary="pad").mean()

taw_coarse.plot.scatter(ax=axes[0,0], x="R̄e_b", y="Γ̄", hue="Fr_h", **Fr_h_cbar_options, **scatter_options, yscale="log")
#taw_coarse.plot.scatter(ax=axes[0,1], x="R̄e_b", y="Γ̄", hue="simulation", **sim_cbar_options, **scatter_options, yscale="log")

taw_coarse.plot.scatter(ax=axes[1,0], x="F̄rₜ", y="Γ̄", hue="Fr_h", **Fr_h_cbar_options, **scatter_options, yscale="log")
#taw_coarse.plot.scatter(ax=axes[1,1], x="F̄rₜ", y="Γ̄", hue="simulation", **sim_cbar_options, **scatter_options, yscale="log")

CSI_coarse.plot.scatter(ax=axes[2,0], x="SP_ratio", y="Γ̄", hue="Fr_h", **Fr_h_cbar_options, **scatter_options)
#CSI_coarse.plot.scatter(ax=axes[2,1], x="SP_ratio", y="Γ̄", hue="simulation", **sim_cbar_options, **scatter_options)
#---

#+++ Adjust panels
Re_b = np.logspace(0, 2.2)
Fr_t = np.logspace(0, 2.2)
for ax in axes.flatten():
    ax.set_title("")
    ax.grid(True)

for ax in axes[0]:
    ax.set_ylim(1e-2, 1)
    ax.plot(Re_b, 8e-1*Re_b**(-1/2), c="k", lw=2, ls="--", label=r"$Re_b^{-1/2}$")
    ax.scatter(taw_Reb_binned.R̄e_b, taw_Reb_binned.Γ̄, s=20, edgecolor="k", c="r")
    ax.legend(loc="upper right")

for ax in axes[1]:
    ax.set_xlim(1e-7, None)
    ax.set_ylim(1e-2, 1)
    ax.plot(Fr_t, 8e-1*Fr_t**(-1), c="k", lw=2, ls="--", label=r"$Fr_t^{-1}$")
    ax.scatter(taw_Frt_binned["F̄rₜ"], taw_Frt_binned.Γ̄, s=20, edgecolor="k", c="r")
    ax.legend(loc="upper right")

for ax in axes[2]:
    #ax.set_xlim(1e-2, 1e2)
    ax.set_ylim(0, 0.5)
    ax.scatter(CSI_SPr_binned.SP_ratio, CSI_SPr_binned.Γ̄, s=20, edgecolor="k", c="r")
#---

#+++ Save figure
fig.savefig(f"figures_check/Reb_dissip{resolution}.png")
#---
