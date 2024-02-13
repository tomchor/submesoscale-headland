import numpy as np
import pynanigans as pn
import xarray as xr
from aux01_physfuncs import get_topography_masks
from aux02_plotting import manual_facetgrid
from dask.diagnostics import ProgressBar
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
plt.rcParams["figure.constrained_layout.use"] = True

#+++ Marker details
marker_large_Bu = "^"
marker_unity_Bu = "s"
marker_small_Bu = "o"

color_large_Bu = "blue"
color_unity_Bu = "orange"
color_small_Bu = "green"
#---

n_cycles = 5
λ_approx = 200
modifiers = ["", "-S"]

for modifier in modifiers:
    etfields = xr.open_dataset(f"data_post/etfields_snaps{modifier}.nc")
    etfields = etfields.reindex(Ro_h = list(reversed(etfields.Ro_h)))
    etfields = etfields.sel(xC=slice(-etfields.Lx/4, None))
    etfields["Bu_h"] = (etfields.Ro_h / etfields.Fr_h)**2

    bulk = xr.open_dataset(f"data_post/bulkstats_snaps{modifier}.nc")
    bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h)))

    #+++ Investigate spatial distribution
    if False:
        for λ in [100]:
            plot_kwargs = dict(x="x", cmap=cm.balance, vmin=-5e-9, vmax=5e-9)
            fig = plt.figure(figsize=(8,8))
            aux, fig = manual_facetgrid(etfields["Π"].mean("time").where(etfields.water_mask_buffered_filt).sel(λ=λ), fig,
                                        land_mask = etfields.land_mask.biject(),
                                        plot_kwargs = plot_kwargs,
                                        add_title=False,
                                        )

            fig = plt.figure(figsize=(8,8))
            etfields["Πᵍ"].where(etfields.dV_water_mask_buffered > 0).mean("time")
            aux, fig = manual_facetgrid(etfields["Πᵍ"].mean("time").where(etfields.water_mask_buffered_filt).sel(λ=λ), fig,
                                        land_mask = etfields.land_mask.biject(),
                                        plot_kwargs = plot_kwargs,
                                        add_title=False,
                                        )

            fig = plt.figure(figsize=(8,8))
            aux, fig = manual_facetgrid(etfields["Πᵃ"].mean("time").where(etfields.water_mask_buffered_filt).sel(λ=λ), fig,
                                        land_mask = etfields.land_mask.biject(),
                                        plot_kwargs = plot_kwargs,
                                        add_title=False,
                                        )
        pause
    #---

    #+++ Get means
    if False:
        etfields["dV"] = (etfields["Δxᶜᶜᶜ"] * etfields["Δyᶜᶜᶜ"] * etfields["Δzᶜᶜᶜ"]).isel(Ro_h=0, Fr_h=0)
        etfields["dV_water_mask_buffered_filt"] = etfields.dV * etfields.water_mask_buffered_filt.isel(Ro_h=0, Fr_h=0)

        def mean(da, dV=etfields["dV_water_mask_buffered_filt"], dims=("x", "y",)):
            return ((da * dV).pnsum(dims) / dV.pnsum(dims)).mean("time")

        print("Averaging energy transfer rates")
        with ProgressBar():
            etfields["Π_mean"]  = mean(etfields["Π"], dims=("x", "y"))#.compute()
            etfields["⟨Πᵍ⟩"] = mean(etfields["Πᵍ"], dims=("x", "y"))#.compute()
            etfields["⟨Πᵃ⟩"] = mean(etfields["Πᵃ"], dims=("x", "y"))#.compute()
    #---

    #+++ Create figure and options
    λ_list = [100, 200, 300]
    size = 3.5
    ncols = len(λ_list)
    print("Creating figures")
    fig, axes = plt.subplots(layout="constrained",
                             ncols = ncols,
                             figsize=(1.3*ncols*size, size),
                             squeeze = False,)
    axesf = axes.flatten()
    #---

    #+++ Plot stuff
    for λ, ax in zip(λ_list, axesf):
        print(f"Plotting results for λ={λ}")
        ax.set_title(f"λ={λ} m")
        #etfields.sel(λ=λ).plot.scatter(ax=ax, x="⟨Πᵍ⟩", y="⟨Πᵃ⟩", marker=marker_large_Bu, edgecolors="none",)

        ds_λ = bulk.sel(λ=λ)
        ax.scatter(ds_λ["⟨Πᵍ⟩"].where(etfields.Bu_h > 1), ds_λ["⟨Πᵃ⟩"].where(etfields.Bu_h > 1),
                   s=60, marker=marker_large_Bu, facecolors=color_large_Bu, edgecolors="none", label=r"$Bu_h>1$")
        ax.scatter(ds_λ["⟨Πᵍ⟩"].where(etfields.Bu_h== 1), ds_λ["⟨Πᵃ⟩"].where(etfields.Bu_h== 1),
                   s=60, marker=marker_unity_Bu, facecolors=color_unity_Bu, edgecolors="none", label=r"$Bu_h=1$")
        ax.scatter(ds_λ["⟨Πᵍ⟩"].where(etfields.Bu_h < 1), ds_λ["⟨Πᵃ⟩"].where(etfields.Bu_h < 1),
                   s=60, marker=marker_small_Bu, facecolors=color_small_Bu, edgecolors="none", label=r"$Bu_h<1$")
    #---

    #+++ Adjust scatterplot panels
    lim = 1e-10
    for ax in axesf:
        ax.grid(True, zorder=-2, linewidth=0.2)
        #ax.set_xlim(-lim, +lim)
        #ax.set_ylim(-lim, +lim)
        ax.set_xscale("symlog", linthresh=lim/100)
        ax.set_yscale("symlog", linthresh=lim/100)
        ax.axhline(y=0, ls="--", color="k", zorder=-1)
        ax.axvline(x=0, ls="--", color="k", zorder=-1)
        ax.set_xlabel("Geostrophic shear production [m²/s³]")
        ax.set_ylabel("Ageostrophic shear production [m²/s³]")

        GSP = np.linspace(-lim/2, +lim/2)
        ax.plot(GSP, -GSP, ls="--", color="gray", zorder=-1)
        ax.legend()

    fig.savefig(f"figures/shearprod{modifier}.pdf", dpi=200)
    #---

