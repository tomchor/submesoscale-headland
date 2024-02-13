import numpy as np
import pynanigans as pn
import xarray as xr
from aux00_utils import filter_by_resolution
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
π = np.pi


#+++ Define directory and simulation name
path = f"./headland_simulations/data/"
simnames = [#"NPN-TEST-f8",
            #"NPN-PropA-f8",
            #"NPN-PropA-f4",
            #"NPN-PropA-f2",
            #"NPN-PropA",
            #"NPN-PropB-f8",
            #"NPN-PropB-f4",
            #"NPN-PropB-f2",
            #"NPN-PropB",
            #"NPN-PropD-f8",
            #"NPN-PropD-f4",
            #"NPN-PropD-f2",
            #"NPN-PropD",
            "NPN-R008F008-f8",
            "NPN-R008F008-f4",
            "NPN-R008F008-f2",
            "NPN-R008F008",
            "NPN-R008F02-f8",
            "NPN-R008F02-f4",
            "NPN-R008F02-f2",
            "NPN-R008F02",
            "NPN-R008F05-f8",
            "NPN-R008F05-f4",
            "NPN-R008F05-f2",
            "NPN-R008F05",
            "NPN-R008F1-f8",
            "NPN-R008F1-f4",
            "NPN-R008F1-f2",
            "NPN-R008F1",
            "NPN-R02F008-f8",
            "NPN-R02F008-f4",
            "NPN-R02F008-f2",
            "NPN-R02F008",
            "NPN-R02F02-f8",
            "NPN-R02F02-f4",
            "NPN-R02F02-f2",
            "NPN-R02F02",
            "NPN-R02F05-f8",
            "NPN-R02F05-f4",
            "NPN-R02F05-f2",
            "NPN-R02F05",
            "NPN-R02F1-f8",
            "NPN-R02F1-f4",
            "NPN-R02F1-f2",
            "NPN-R02F1",
            "NPN-R05F008-f8",
            "NPN-R05F008-f4",
            "NPN-R05F008-f2",
            "NPN-R05F008",
            "NPN-R05F02-f8",
            "NPN-R05F02-f4",
            "NPN-R05F02-f2",
            "NPN-R05F02",
            "NPN-R05F05-f8",
            "NPN-R05F05-f4",
            "NPN-R05F05-f2",
            "NPN-R05F05",
            "NPN-R05F1-f8",
            "NPN-R05F1-f4",
            "NPN-R05F1-f2",
            "NPN-R05F1",
            "NPN-R1F008-f8",
            "NPN-R1F008-f4",
            "NPN-R1F008-f2",
            "NPN-R1F008",
            "NPN-R1F02-f8",
            "NPN-R1F02-f4",
            "NPN-R1F02-f2",
            "NPN-R1F02",
            "NPN-R1F05-f8",
            "NPN-R1F05-f4",
            "NPN-R1F05-f2",
            "NPN-R1F05",
            "NPN-R1F1-f8",
            "NPN-R1F1-f4",
            "NPN-R1F1-f2",
            "NPN-R1F1",
            ]
#---

def params_background(ax, add_lines=True, cmap="Blues"):
    arr = np.logspace(-1.5, .5, 200)
    Ro_h = xr.DataArray(arr, dims="Ro_h", coords=dict(Ro_h=arr))
    Fr_h = xr.DataArray(arr, dims="Fr_h", coords=dict(Fr_h=arr))
    Bu_h = (Ro_h / Fr_h)**2

    Bu_h.name = r"$Bu_h = \left( Ro_h / Fr_h \right)^2$"

    Bu_h.plot.contourf(ax=ax, x="Ro_h", levels=np.logspace(-4, 4, 9), norm=LogNorm(),
                       xscale="log", yscale="log", cmap=cmap,
                       cbar_kwargs = dict(location="bottom",
                                          fraction=0.08,
                                          pad=0.05,
                                          shrink=0.9))
    if add_lines:
        ax.plot(Fr_h, Fr_h * np.sqrt(12.), ls="--", c="k")
        ax.plot(Fr_h, Fr_h * np.sqrt(5.5), ls="-", c="k")


n_cycles = 5
L_approx = [20, 50, 100, 200]
resolutions = ["f4", "f2", ""]
resolutions = ["f2",]
for resolution in resolutions:
    simnames_filtered = [ simname for simname in simnames if filter_by_resolution(simname, resolution) ]

    #+++ Create figure and options
    fig, ax = plt.subplots(layout="constrained", figsize=(5, 5))
    params_background(ax, add_lines=False, cmap="Blues")
    #---

    for sim_number, simname in enumerate(simnames_filtered):
        print(f"Opening {simname}")
        etfields = xr.open_dataset(f"data_post/etfields_{simname}.nc", chunks=dict(time="auto", L="auto"))
        ax.scatter(etfields.Ro_h, etfields.Fr_h, label=simname[4:], s=120, color="black", zorder=10)

    #+++ Include secondary x axis (f₀)
    Ro2f = lambda x: x * (etfields.V_inf/(etfields.Lx/4))
    f2Ro = lambda x: (etfields.V_inf/(etfields.Lx/4)) / x
    secxax = ax.secondary_xaxis("top", functions=(Ro2f, f2Ro))
    secxax.set_xlabel("f₀")
    #---

    #+++ Include secondary y axis (N∞)
    Fr2N = lambda x: x * (etfields.V_inf/etfields.H)
    N2Fr = lambda x: (etfields.V_inf/etfields.H) / x
    secyax = ax.secondary_yaxis("right", functions=(Fr2N, N2Fr))
    secyax.set_ylabel("N∞")
    #---

    #+++ Adjust panel
    ax.grid(True, zorder=0)
    ax.set_title("Large-Eddy Simulations")

    ax.set_xlabel(r"$Ro_h = V_\infty / L f_0$")
    ax.set_ylabel(r"$Fr_h = V_\infty / H N_\infty$")
    #---

    #+++ Save
    fig.savefig(f"figures_check/paramspace.pdf")
    #---
