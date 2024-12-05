import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import xarray as xr
import pynanigans as pn
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#+++ Define directory and simulation name
path = f"./headland_simulations/data/"
simnames = ["NPN-R008F008",
            "NPN-R008F02",
            "NPN-R008F05",
            "NPN-R008F1",
            "NPN-R02F008",
            "NPN-R02F02",
            "NPN-R02F05",
            "NPN-R02F1",
            "NPN-R05F008",
            "NPN-R05F02",
            "NPN-R05F05",
            "NPN-R05F1",
            "NPN-R1F008",
            "NPN-R1F02",
            "NPN-R1F05",
            "NPN-R1F1",
            ]

from cycler import cycler
names = cycler(name=simnames)
modifiers = cycler(modifier = ["-f2", ""])
simnames = [ nr["name"] + nr["modifier"] for nr in modifiers * names ]
#---

fpaths = [ f"data_post/bulkstats_{simname}.nc" for simname in simnames ]
bulk_stats = xr.open_mfdataset(fpaths, concat_dim="simulation", combine="nested")

bulk_stats["Δz̃ᵋ"] = bulk_stats["Δz_min"] / bulk_stats["Loᵋ"]
bulk_stats["Δz̃ᵋ"].attrs = dict(long_name="$\Delta z / L_O$", units="")
bulk_stats["Δz_min"].attrs = dict(long_name="$\Delta z$", units="m")
bulk_stats["Slope_Bu"].attrs = dict(long_name="Slope Burger number $S_h$", units="")

bulk_stats = bulk_stats.rename(Fr_h = "Frₕ")

fig, ax = plt.subplots(figsize=(5,4), constrained_layout=True)
bulk_stats.plot.scatter(x="Slope_Bu", y="Δz̃ᵋ", hue="Δz_min",
                        s=60, xscale="log",
                        add_legend=True, add_colorbar=False,
                        cmap = LinearSegmentedColormap.from_list("binary", ["black", "lightgray"]),
                        edgecolors = "none",
                        rasterized = True,
                        )
ax = plt.gca()
ax.axhline(y=1, ls="--", color="k")
ax.set_title("Vertical spacing / Ozmidov scale")
ax.grid(True)

fig.savefig(f"figures/resolvedness.pdf")
