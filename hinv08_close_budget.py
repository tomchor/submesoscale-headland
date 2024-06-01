import sys
sys.path.append("/glade/u/home/tomasc/repos/pynanigans")
from os.path import basename
import numpy as np
import xarray as xr
import pynanigans as pn
from aux00_utils import open_simulation
from matplotlib import pyplot as plt
plt.rcParams['figure.constrained_layout.use'] = True
π = np.pi

print("Starting bulk statistics script")

#+++ Define directory and simulation name
if basename(__file__) != "h00_runall.py":
    path = f"./headland_simulations/data/"
    simnames = [#"NPN-TEST",
                #"NPN-R008F008",
                #"NPN-R02F008",
                #"NPN-R05F008",
                #"NPN-R1F008",
                #"NPN-R008F02",
                "NPN-R02F02",
                "NPN-R05F02",
                "NPN-R1F02",
                "NPN-R008F05",
                "NPN-R02F05",
                "NPN-R05F05",
                "NPN-R1F05",
                "NPN-R008F1",
                "NPN-R02F1",
                "NPN-R05F1",
                "NPN-R1F1",
                ]

    from cycler import cycler
    names = cycler(name=simnames)
    modifiers = cycler(modifier = ["-f4", "-S-f4", "-f2", "-S-f2", "", "-S"])
    modifiers = cycler(modifier = ["-f4",])
    simnames = [ nr["name"] + nr["modifier"] for nr in modifiers * names ]
#---

outnames = []
for simname in simnames:
    #+++ Open datasets
    print(f"\nOpening {simname}")
    grid_xyi, xyi = open_simulation(path+f"xyi.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks=dict(time="auto")),
                                    )
    grid_xyz, xyz = open_simulation(path+f"xyz.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks=dict(time="auto")),
                                    )
    tafields = xr.open_dataset(f"data_post/tafields_{simname}.nc", decode_times=False, chunks="auto")
    #---

    #+++ Calculate auxiliary variables
    tafields["∫∫∫ᵇuᵢGᵢ²dxdydz"] = -tafields["∫∫∫ᵇ⟨uᵢ∂ⱼuⱼuᵢ⟩ₜdxdydz"]\
                                  +tafields["∫∫∫ᵇ⟨wb⟩ₜdxdydz"]\
                                  +tafields["∫∫∫ᵇε̄ₛdxdydz"]\
                                  -tafields["∫∫∫ᵇ⟨uᵢ∂ⱼτᵢⱼ⟩ₜdxdydz"]\
                                  -tafields["∫∫∫ᵇ⟨uᵢ∂ⱼτᵇᵢⱼ⟩ₜdxdydz"]
    tafields["∫∫∫ᵇ⟨∂ⱼ(uᵢτᵢⱼ)⟩ₜdxdydz"] = tafields["∫∫∫ᵇ⟨uᵢ∂ⱼτᵢⱼ⟩ₜdxdydz"] - tafields["∫∫∫ᵇε̄ₖdxdydz"]
    #---

    for buffer in [0, 5,]:
        #+++
        fig, axes = plt.subplots(ncols=1, figsize=(4, 8))
        ax = axes

        term_names = ("-∂ₜk²", "-uᵢ∂ⱼ(uⱼuᵢ)", "-∂ⱼ(uⱼp)", "wb", "uᵢFᵢ", "-∂ⱼ(uᵢτᵢⱼ)", "-εₖ", "-uᵢ∂ⱼτᵇᵢⱼ", "Residual")
        y_pos = np.arange(len(term_names))
        term_values = [-tafields["∫∫∫ᵇ⟨∂ₜEk⟩ₜdxdydz"].sel(buffer=buffer),
                       -tafields["∫∫∫ᵇ⟨uᵢ∂ⱼuⱼuᵢ⟩ₜdxdydz"].sel(buffer=buffer),
                       -tafields["∫∫∫ᵇ⟨uᵢ∂ᵢp⟩ₜdxdydz"].sel(buffer=buffer),
                       +tafields["∫∫∫ᵇ⟨wb⟩ₜdxdydz"].sel(buffer=buffer),
                       +tafields["∫∫∫ᵇε̄ₛdxdydz"].sel(buffer=buffer),
                       -tafields["∫∫∫ᵇ⟨∂ⱼ(uᵢτᵢⱼ)⟩ₜdxdydz"].sel(buffer=buffer),
                       -tafields["∫∫∫ᵇε̄ₖdxdydz"].sel(buffer=buffer),
                       -tafields["∫∫∫ᵇ⟨uᵢ∂ⱼτᵇᵢⱼ⟩ₜdxdydz"].sel(buffer=buffer),
                       ]
        term_values.append(sum(term_values))

        ax.barh(y_pos, term_values, align="center")
        ax.set_yticks(y_pos, labels=term_names)
        ax.invert_yaxis()  # labels read top-to-bottom
        ax.set_xlabel("Term contribution")
        ax.set_title("Term-by-term budget")
        #---

        #+++ Prettify and save
        for ax in [axes]:
            ax.axvline(x=0, ls="--", color="k")
            ax.grid(True)
        fig.savefig(f"figures_check/budget_{simname}_buffer={buffer}m.png")
        #---
    pause
