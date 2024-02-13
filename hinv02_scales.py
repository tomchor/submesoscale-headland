import numpy as np
import pynanigans as pn
import xarray as xr
from aux00_utils import open_simulation
from dask.diagnostics import ProgressBar
π = np.pi

#+++ Define directory and simulation name
path = f"./headland_simulations/data/"
simnames = [#"NPN-R1F02A01",
            #"NPN-TEST-f8",
            #"NPN-TEST-f8",
            #"NPN-TEST-f2",
            #"NPN-TEST",
            #"NPN-PropA-f8",
            #"NPN-PropA-f4",
            #"NPN-PropA-f2",
            #"NPN-PropA",
            #"NPN-PropB-f8",
            #"NPN-PropB-f4",
            "NPN-PropB-f2",
            #"NPN-PropB",
            #"NPN-PropD-f8",
            #"NPN-PropD-f4",
            #"NPN-PropD-f2",
            #"NPN-PropD",
            #"NPN-R005F002-f4",
            #"NPN-R005F002-f2",
            #"NPN-R005F02-f4",
            #"NPN-R005F02-f2",
            #"NPN-R005F03-f4",
            #"NPN-R005F03-f2",
            #"NPN-R008F02-f4",
            #"NPN-R008F02-f2",
            #"NPN-R008F03-f4",
            #"NPN-R008F03-f2",
            #"NPN-Magaldi4h-f4",
            #"NPN-Magaldi4h-f2",
            #"NPN-Magaldi6h-f4",
            #"NPN-Magaldi6h-f2",
            #"NPN-Magaldi6h_steep-f4",
            #"NPN-Magaldi6h_steep-f2",
            ]
#---

size = 3.5
n_cycles = 10
dslist = []
for simname in simnames:
    fields = xr.open_dataset(f"data_post/fields_{simname}.nc")

    for var in ["Uᵍₖ", "Ũᵍₖ", "Uᵍₖ′"]:
        fields[var] = fields[var].rename(k="i")
        fields[var].loc[dict(i=2)] += fields.V_inf

    fields = fields.sel(i=[1,2])
    opts = dict(vmin=-1e-2, vmax=+1e-2, cmap="RdBu_r")
#    for var in ["Ũᵍₖ", "Uᵍₖ′", "uᵢ̃", "uᵢ′"]:
#        fields[var].isel(time=-1, z=1).sel(y=slice(-500, 2000)).pnplot(y="y", col="L", row="i", **opts)
    for var in ["Uᵍₖ", "uᵢ"]:
        fields[var].isel(time=-1, z=1).sel(y=slice(-500, 2000)).pnplot(y="y", row="i", **opts)

