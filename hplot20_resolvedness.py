import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
import numpy as np
import xarray as xr
import pynanigans as pn
from matplotlib import pyplot as plt

#+++ Define directory and simulation name
path = f"./headland_simulations/data/"
simnames = [#"NPN-TEST",
            #"NPN-PropA",
            #"NPN-PropB",
            #"NPN-PropD",
            "NPN-R008F008",
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
modifiers = cycler(modifier = ["-f4", "-f2", ""])
simnames = [ nr["name"] + nr["modifier"] for nr in modifiers * names ]
#---

fpaths = [ f"data_post/bulkstats_{simname}.nc" for simname in simnames ]
bulk_stats = xr.open_mfdataset(fpaths, concat_dim="simulation", combine="nested")
bulk_stats["Δz̃ᵋ"] = bulk_stats["Δz_min"] / bulk_stats["Loᵋ"]

bulk_stats["Δz̃ᵋ"].attrs = dict(long_name="$\Delta z / L_O$ᵋ", units="")

bulk_stats = bulk_stats.rename(Fr_h = "Frₕ")

fig = plt.figure(figsize=(5,4))
bulk_stats.plot.scatter(y="Δz̃ᵋ", x="Δz_min", facecolors="black", edgecolors="none")

ax = plt.gca()
ax.set_xlabel("Δz [m]")
ax.set_ylabel("$\Delta z / L_O$")
ax.axhline(y=1, ls="--", color="k")
ax.set_title("Vertical spacing / Ozmidov scale")
ax.grid(True)

fig.savefig(f"figures_check/resolvedness.pdf")
