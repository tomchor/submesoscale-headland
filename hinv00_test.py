import numpy as np
import xarray as xr
from cmocean import cm
from matplotlib import pyplot as plt
from aux00_utils import open_simulation
import pynanigans as pn
plt.rcParams['figure.constrained_layout.use'] = True


#+++ Define dir and file names
path = f"./headland_simulations/data/"
simnames = [#"NPN-TEST",
            #"NPN-R008F008",
            #"NPN-R008F02",
            #"NPN-R008F05",
            #"NPN-R008F1",
            #"NPN-R02F008",
            #"NPN-R02F02",
            #"NPN-R02F05",
            #"NPN-R02F1",
            #"NPN-R05F008",
            #"NPN-R05F02",
            "NPN-R05F05",
            #"NPN-R05F1",
            #"NPN-R1F008",
            #"NPN-R1F02",
            #"NPN-R1F05",
            #"NPN-R1F1",
            ]

from cycler import cycler
names = cycler(name=simnames)
modifiers = cycler(modifier = ["-f4"])
simnames = [ nr["name"] + nr["modifier"] for nr in modifiers * names ]
#---

for simname in simnames:
   
    #+++ Open datasets xyz and xyi
    print(f"\nOpening {simname} xyz")
    grid_xyz, xyz = open_simulation(path+f"xyz.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks="auto"),
                                    )
    print(f"Opening {simname} xyi")
    grid_xyi, xyi = open_simulation(path+f"xyi.{simname}.nc",
                                    use_advective_periods=True,
                                    topology=simname[:3],
                                    squeeze=True,
                                    load=False,
                                    open_dataset_kwargs=dict(chunks="auto"),
                                    )
    #---

    xyz = xyz.isel(time=-1)
    xyi = xyi.isel(time=-1)

    dA = xyz.ΔxΔz.where((xyz.zC>xyz.H/2) & (xyz.zC <3*xyz.H/2), other=0)
    dV = (xyz.ΔxΔz * xyz["Δyᶜᶜᶜ"]).where((xyz.zC>xyz.H/2) & (xyz.zC <3*xyz.H/2), other=0)

    prefix_map = { 0:"∫∫⁰", 5:"∫∫⁵", 10:"∫∫¹⁰", 20:"∫∫²⁰"}
    for buffer in xyz.buffers:
        title = f"Buffer = {buffer} meters"
        print(title)
        prefix = prefix_map[buffer]
        volume_2d = f"{prefix}dxdz"
        varname_2d = f"{prefix}εₖdxdz"
        xyz[varname_2d] = (xyz["εₖ"]*dA.where(xyz.altitude > buffer)).pnsum(("x", "z"))

        volume_3d = f"∫{prefix}dxdydz"
        varname_3d = f"{prefix}εₖdxdydz"
        xyz[varname_3d] = (xyz["εₖ"]*dV.where(xyz.altitude > buffer)).pnsum(("x", "z"))

        plt.figure()
        (xyi[varname_2d] / xyi[volume_2d]).pnplot(x="y", color="C0", label="online")
        (xyz[varname_2d] / xyi[volume_2d]).pnplot(x="y", color="C1", label="offline", ls="--")

        online_volume_average = xyi[varname_3d] / xyi[volume_3d]
        offline_volume_average = (xyi[varname_2d] * xyi["Δyᶜᶜᶜ"].max("xC")).sum() / xyi[volume_3d].sum()

        plt.axhline(y=online_volume_average, color = "C0")
        plt.axhline(y=offline_volume_average, color = "C1", ls="--")
        plt.title(title); plt.legend()
