import numpy as np
import pynanigans as pn
import xarray as xr
from aux00_utils import open_simulation
from matplotlib import pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True

modifier = ""
summarize = True

#+++ Open and unify datasets
tafields = xr.open_dataset(f"data_post/tafields_snaps{modifier}.nc")

#snaps = xr.open_dataset(f"data_post/xyi_snaps{modifier}.nc")[["PV_norm", "PV_x", "PV_y", "PV_z", "dbdz", "N²∞", "f₀", "Ro", "Ri"]]
snaps = xr.open_dataset(f"data_post/xyi_snaps{modifier}.nc")[["PV_norm", "dbdz", "N²∞", "f₀", "Ro", "Ri"]]
snaps = snaps.sel(yC=slice(snaps.yC[0], np.inf))

snaps = snaps.reindex_like(tafields)
snaps["q̃"] = tafields["q̃"]
#snaps = snaps.where(tafields.water_mask_buffered)
#---

#+++ Pre-processing before plotting
if summarize:
    summary_values = [0.2, 1.25]
    snaps = snaps.sel(Ro_h = summary_values, Fr_h = summary_values)

snaps = snaps.reindex(Ro_h = list(reversed(snaps.Ro_h)))
snaps.Ro_h.attrs = dict(long_name=r"Roₕ")
snaps.Fr_h.attrs = dict(long_name=r"Frₕ")
snaps.PV_norm.attrs = dict(long_name=r"Normalized Ertel PV")
snaps = snaps.sel(time=slice(None, None, 20), xC=slice(0, np.inf), yC=slice(0, np.inf))
#---


#+++ Create relevant variables
#+++ dbdz_norm
snaps["dbdz_norm"] = snaps.dbdz / snaps["N²∞"]
snaps.dbdz_norm.attrs = dict(long_name = "$\partial b/\partial z$ (normalized)")
#---

snaps.PV_norm.attrs = dict(long_name = "PV (normalized)")

snaps["q̃_norm"] = snaps["q̃"]/ (snaps["N²∞"] * snaps["f₀"])
snaps["q̃_norm"].attrs = dict(long_name = "Filtered PV (normalized)")

snaps = snaps.reindex(λ=np.append(0, snaps.λ))
snaps["q̃_norm"][dict(λ=0)] = snaps.PV_norm.transpose(*snaps["q̃_norm"].sel(λ=0).dims)

#+++ thomas angle-related stuff
#snaps["PVz_norm"] = snaps.PV_z / (snaps["N²∞"] * snaps["f₀"])
#snaps["PVh_norm"] = (snaps.PV_x + snaps.PV_y) / (snaps["N²∞"] * snaps["f₀"])
snaps["1/Ri"] = 1/snaps.Ri
snaps["Ro_tot"] = snaps.Ro + 1
snaps.Ro_tot.attrs = dict(long_name="1 + Ro")
#---
#---

#+++ Plot options
cbar_kwargs = dict(orientation="vertical", location="right", shrink=0.5, extend="both")
scatter_options = dict(s=4, hue="Ro_tot", vmin=-3, vmax=+3, cmap="coolwarm", rasterized=True)
#---

#+++ Phase diagram with dbdz
if True:
    for λ in snaps.λ.values[-1:]:
        print(f"Plotting scatterplot for λ={λ}")
        fg = snaps.sel(λ=λ).where(tafields.water_mask_buffered).plot.scatter(x="q̃_norm", y="dbdz_norm", col="Fr_h", row="Ro_h",
                                                                             sharex = False, sharey = False,
                                                                             edgecolors="none", cbar_kwargs=cbar_kwargs,
                                                                             **scatter_options)

        for ax in fg.axs.flat:
            ax.grid(True)
            ax.axhline(y=0, ls="--", color="black")
            ax.axvline(x=0, ls="--", color="black")

            xdata, ydata = np.array(ax.collections[0].get_offsets()).T
            ax.set_xlim(min(-0.2, np.nanmin(xdata)), np.nanmax(xdata))
            ax.set_ylim(np.nanmin(ydata), np.nanmax(ydata))

        #+++ Save figure
        print("Saving fig")
        plt.gcf().savefig(f"figures/phase-diagram_dbdz_λ={λ}{modifier}.pdf")
        #---
#---

#+++ Phase diagram with Ri
if False:
    fg = snaps.where(tafields.water_mask_buffered).plot.scatter(x="q̃_norm", y="Ri", col="Fr_h", row="Ro_h",
                                                                sharex = False, sharey = False,
                                                                edgecolors="none", cbar_kwargs=cbar_kwargs,
                                                                **scatter_options)
    for ax in fg.axs.flat:
        ax.grid(True)
        ax.axhline(y=0, ls="--", color="red")
        ax.axvline(x=0, ls="--", color="red")
    
        xdata, ydata = np.array(ax.collections[0].get_offsets()).T
        ax.set_xlim(np.nanmin(xdata), np.nanmax(xdata))
        ax.set_ylim(np.nanmin(ydata), 1e3)
    
    #+++ Save figure
    print("Saving fig")
    plt.gcf().savefig(f"figures/phase-diagram_Ri.pdf")
    #---
#---


