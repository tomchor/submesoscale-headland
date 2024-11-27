import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm
from aux02_plotting import BuRd, letterize
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

xyz = xr.open_dataset("headland_simulations/data/xyz.NPN-R1F008.nc", decode_times=False).squeeze()
xyz = xyz.assign_coords(time=xyz.time/xyz.T_advective)
xyz = xyz.sel(time=30, xC=slice(-100, None), yC=slice(0, None)).sel(zC=40, method="nearest")

Ri_left = -2; Ri_right = 4
Ri_bins = np.linspace(Ri_left, Ri_right, 50)
Ri_bin_labels = (Ri_bins[1:] + Ri_bins[:-1]) / 2

xyz_Ri_binned = xyz.groupby_bins("Ri", Ri_bins, labels=Ri_bin_labels).mean()

fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True, sharey=True, figsize=(6, 5))

ax.scatter(xyz_Ri_binned.Ri, xyz_Ri_binned["κₑ"], color="black")
ax.set_xlim(Ri_left, Ri_right)
ax.set_ylim(0, None)

Ri = np.linspace(Ri_left, Ri_right, 200)
ν0 = float(max(xyz_Ri_binned["κₑ"])) # 50e-4 # m²/s
Ri0_list = [0.8, 1.5, 3]
for Ri0 in Ri0_list:
    νt = ν0 * (1 - (Ri/Ri0)**2)**3
    ax.plot(Ri, νt, label=f"$Ri_0 =$ {Ri0}")
ax.legend()
