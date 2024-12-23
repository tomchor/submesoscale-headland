import numpy as np
import pynanigans as pn
import xarray as xr
from aux00_utils import filter_by_resolution, threed_sims, bathfo_sims, vertco_sims, vertsh_sims
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
π = np.pi


#+++ Define directory and simulation name
path = f"./headland_simulations/data/"
simnames_base = [#"NPN-TEST",
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
modifiers = ["-f4",]
#---

#+++ Define function to create the background to the parameter space
def params_background(ax, add_lines=True, cmap="Blues"):
    arr = np.logspace(-1.5, .5, 200)
    Ro_h = xr.DataArray(arr, dims="Ro_h", coords=dict(Ro_h=arr))
    Fr_h = xr.DataArray(arr, dims="Fr_h", coords=dict(Fr_h=arr))
    Bu_h = (Ro_h / Fr_h)**2
    S_h = Ro_h / Fr_h

    Ro_h.name = r"$Bu_h$"
    Fr_h.name = r"$Fr_h$"
    Bu_h.name = r"$Bu_h = \left( Ro_h / Fr_h \right)^2$"
    S_h.name = r"$S_h = Ro_h / Fr_h = H N_\infty / L f_0$"

    S_h.plot.contourf(ax=ax, x="Fr_h", levels=np.logspace(-2, 2, 9), norm=LogNorm(),
                      xscale="log", yscale="log", cmap=cmap, extend="both",
                      cbar_kwargs = dict(location="bottom",
                                         fraction=0.05,
                                         pad=0.04,
                                         shrink=0.9,),)
    if add_lines:
        ax.plot(Fr_h, Fr_h * np.sqrt(12.), ls="--", c="k")
        ax.plot(Fr_h, Fr_h * np.sqrt(5.5), ls="-", c="k")
#---

for modifier in modifiers:
    simnames = [ simname_base + modifier for simname_base in simnames_base ]

    #+++ Create figure and options
    fig, ax = plt.subplots(layout="constrained", figsize=(5, 7))
    params_background(ax, add_lines=False, cmap="Blues")
    #---

    for sim_number, simname in enumerate(simnames):
        print(f"Opening {simname}")
        xyi = xr.open_dataset(path+f"/xyi.{simname}.nc", decode_times=False)
        if (np.round(xyi.Ro_h, decimals=2) in [0.2, 1.25]) and (np.round(xyi.Fr_h, decimals=2) in [0.2, 1.25]):
            ax.scatter(xyi.Fr_h, xyi.Ro_h, label=simname[4:], s=300, edgecolor="red", color=[0,0,0,0], zorder=10, linewidths=3)

        if simname.split("-")[1] in threed_sims:
            p3d = ax.scatter(xyi.Fr_h, xyi.Ro_h, s=120, marker="^", color="green", zorder=10, label="Small-scale turb.")
        elif simname.split("-")[1] in bathfo_sims:
            pbf = ax.scatter(xyi.Fr_h, xyi.Ro_h, s=120, marker="X", color="blue", zorder=10, label="Bath. following")
        elif simname.split("-")[1] in vertco_sims:
            pvc = ax.scatter(xyi.Fr_h, xyi.Ro_h, s=120, marker="D", color="orange", zorder=10, label="Vert. coupled")
        elif simname.split("-")[1] in vertsh_sims:
            pvs = ax.scatter(xyi.Fr_h, xyi.Ro_h, s=120, marker="d", color="orchid", zorder=10, label="Vert. decoupled")
        else:
            ptr = ax.scatter(xyi.Fr_h, xyi.Ro_h, s=120, marker="o", color="black", zorder=10, label="Transition regime")

    #+++ Include secondary x axis (f₀)
    Ro2f = lambda x: x * (xyi.V_inf/(xyi.Lx/4))
    f2Ro = lambda x: (xyi.V_inf/(xyi.Lx/4)) / x
    secyax = ax.secondary_yaxis("right", functions=(Ro2f, f2Ro))
    secyax.set_ylabel(r"$f_0$ [1/s]")
    #---

    #+++ Include secondary y axis (N∞)
    Fr2N = lambda x: x * (xyi.V_inf/xyi.H)
    N2Fr = lambda x: (xyi.V_inf/xyi.H) / x
    secxax = ax.secondary_xaxis("top", functions=(Fr2N, N2Fr))
    secxax.set_xlabel(r"$N_\infty$ [1/s]")
    #---

    #+++ Adjust panel
    ax.grid(True, zorder=0)
    ax.legend((p3d, pbf, pvc, pvs, ptr),
              ("Small-scale turbulence", "Bathymetry-following", "Vertically-coupled eddying", "Vertically-decoupled eddying", "Transitional"),
              numpoints=1,
              loc="lower center", bbox_to_anchor=(0.5, 1.15))
    ax.set_aspect("equal")
    #ax.set_title("Large-Eddy Simulations")

    ax.set_xlabel(r"$Fr_h = V_\infty / H N_\infty$")
    ax.set_ylabel(r"$Ro_h = V_\infty / L f_0$")
    #---

    #+++ Save
    fig.savefig(f"figures/paramspace.pdf")
    #---
