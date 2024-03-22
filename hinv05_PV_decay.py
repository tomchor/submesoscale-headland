import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from scipy.optimize import curve_fit
from aux02_plotting import letterize

modifier = ""
tafields = xr.open_dataset(f"data_post/tafields_snaps{modifier}.nc", chunks={})
tafields = tafields.sel(yC=slice(-tafields.L/2, np.inf))
bulk = xr.open_dataset(f"data_post/bulkstats_snaps{modifier}.nc", chunks={})
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h)))

tafields["Slope_Bu"] = tafields.Ro_h / tafields.Fr_h
bulk.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
bulk.yC.attrs =  dict(long_name=r"$y$", units="m")
q̂_min = (tafields.q̄.sel(yC=slice(-tafields.L/2, +tafields.L/2)).pnmin(("x", "y")) / (tafields["f₀"] * tafields["N²∞"])) # Close to headland tip
q̂_minx = tafields.q̄.where(tafields.average_CSI_mask).pnmin("x") / (tafields["f₀"] * tafields["N²∞"])
q̂_mean = tafields.q̄.where(tafields.average_CSI_mask).pnmean("x") / (tafields["f₀"] * tafields["N²∞"])

#+++ Create figure
nrows = 2
ncols = 2
size = 3
fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                         figsize = (2*ncols*size, nrows*size),
                         sharex = "col", sharey = False,
                         squeeze = False,
                         constrained_layout=True)
axesf = axes.flatten()
#---

cmap = plt.cm.coolwarm
for Ro_h in tafields.Ro_h:
    for Fr_h in tafields.Fr_h:
        tafields_RF = tafields.sel(Ro_h=Ro_h, Fr_h=Fr_h)
        q̂_minx_RF = q̂_minx.sel(Ro_h=Ro_h, Fr_h=Fr_h)
        q̂_mean_RF = q̂_mean.sel(Ro_h=Ro_h, Fr_h=Fr_h)

        #+++ Get distance norm
        q̂_min0 = q̂_min.sel(Ro_h=Ro_h, Fr_h=Fr_h)
        S_normalized = (np.log10(tafields_RF.Slope_Bu) - np.log10(tafields.Slope_Bu).min()) / (2*np.log10(tafields.Slope_Bu).max())
        L_C = tafields_RF.V_inf * np.sqrt(-q̂_min0) / tafields_RF["f₀"]
        y_norm = tafields_RF.yC.values / L_C.values
        #---

        if (tafields_RF.Slope_Bu>1):
            ax1 = axes[0,0]
            ax2 = axes[1,0]
        elif (tafields_RF.Slope_Bu<1):
            if ncols >= 2:
                ax1 = axes[0,1]
                ax2 = axes[1,1]
            else:
                continue
        else:
            continue

        (q̂_minx_RF/q̂_minx_RF.sel(yC=0, method="nearest")).assign_coords(yC=y_norm).pnplot(ax=ax1, x="y", color=cmap(S_normalized))
        ax1.set_ylabel("q_min normalized")

        (q̂_mean_RF/q̂_mean_RF.sel(yC=0, method="nearest")).assign_coords(yC=y_norm).pnplot(ax=ax2, x="y", color=cmap(S_normalized))
        ax2.set_ylabel("q_mean normalized")

norm = LogNorm(vmin=tafields.Slope_Bu.min().values, vmax=tafields.Slope_Bu.max().values)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
plt.colorbar(sm, ax=ax2, label = "Slope Burger number")

#+++ Prettify and save
for ax_row in axes:
    ax = ax_row[0]
    ax.set_xlim(0, 1)

    for ax in ax_row:
        ax.grid(axis="y")
        ax.set_title(f"")
        ax.axvline(x=0, color="lightgray", ls="--", lw=1, zorder=0)
letterize(axesf, x=0.05, y=0.9, fontsize=14)

pause
fig.savefig(f"figures_check/PV_decay_buffer={buffer}m{modifier}.pdf")
#---

