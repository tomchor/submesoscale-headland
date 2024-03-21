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
bulk = xr.open_dataset(f"data_post/bulkstats_snaps{modifier}.nc", chunks={})
bulk = bulk.reindex(Ro_h = list(reversed(bulk.Ro_h)))

variables = ["⟨ε̄ₖ⟩ˣᶻ", "⟨ε̄ₚ⟩ˣᶻ"]
bulk["⟨ε̄ₖ⟩ˣᶻ"].attrs = dict(units="m²/s³")
bulk["⟨ε̄ₚ⟩ˣᶻ"].attrs = dict(units="m²/s³")
bulk.Slope_Bu.attrs =  dict(long_name=r"$S_{Bu} = Bu_h^{1/2} = Ro_h / Fr_h$")
bulk.yC.attrs =  dict(long_name=r"$y$", units="m")

#+++ Bathymetry intrusion exponential
def η(z): return bulk.Lx/2 + (0 - bulk.Lx/2) * z / (2*bulk.H) # headland intrusion size
def headland_x_of_yz(y, z=40): return η(z) * np.exp(-(2*y / η(z))**2)
#---

for buffer in bulk.buffer.values:
    print(f"Plotting buffer = {buffer} m")
    bulkb = bulk.sel(buffer=buffer)
    #+++ Create figure
    nrows = len(variables)
    ncols = 1
    size = 3
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows,
                             figsize = (2*ncols*size, nrows*size),
                             sharex = True, sharey = True,
                             squeeze = False,
                             constrained_layout=True)
    axesf = axes.flatten()
    #---

    cmap = plt.cm.copper_r
    cmap = plt.cm.coolwarm
    for Ro_h in bulkb.Ro_h:
        for Fr_h in bulkb.Fr_h:
            bulk0 = bulkb.sel(Ro_h=Ro_h, Fr_h=Fr_h)
            S_normalized = (np.log10(bulk0.Slope_Bu) - np.log10(bulkb.Slope_Bu).min()) / (2*np.log10(bulkb.Slope_Bu).max())
            for ax, variable in zip(axesf, variables):
                bulk0[variable].pnplot(ax=ax, x="y", color=cmap(S_normalized))

    norm = LogNorm(vmin=bulkb.Slope_Bu.min().values, vmax=bulkb.Slope_Bu.max().values)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label = "Slope Burger number")

    #+++ Prettify and save
    for ax in axesf:
        ax.grid(axis="y")
        ax.axvline(x=0, color="lightgray", ls="--", lw=1, zorder=0)
        ax.set_xlim(-250, bulk.yC[-1])
        ax.set_ylim(0, None)
        ax.set_title(f"")
        #ax.set_title(f"Average of KE dissipation rate\nexcluding {bulkb.buffer.item()} m closest to the boundary")

        ax2 = ax.twinx()
        ax2.fill_between(bulkb.yC, headland_x_of_yz(bulkb.yC), color="lightgray", alpha=.8)
        ax2.set_ylim(0, 2e3)
        ax2.tick_params(left=False, right=False, bottom=False, labelleft=False, labelright=False, labelbottom=False)
    letterize(axesf, x=0.05, y=0.9, fontsize=14)

    fig.savefig(f"figures/dissipation_curves_buffer={buffer}m{modifier}.pdf")
    #---