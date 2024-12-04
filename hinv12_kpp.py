import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm
from aux00_utils import open_simulation
from aux02_plotting import BuRd, letterize
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

simname = "R1F008"
xyz = open_simulation(f"headland_simulations/data/xyz.NPN-{simname}.nc",
                      use_advective_periods=True,
                      decode_times=False,
                      get_grid=False)
xyz = xyz.sel(xC=slice(-100, None), yC=slice(0, None)).sel(time=30, zC=40, method="nearest")

Ri_left = -2; Ri_right = 4
Ri_bins = np.linspace(Ri_left, Ri_right, 50)
Ri_bin_labels = (Ri_bins[1:] + Ri_bins[:-1]) / 2

xyz["ν_scaled"] = xyz["κₑ"] * 10 * (10**0.85)# This comes from νₑ ~ κₑ ~ V∞ L, assuming we scale L by 10 times
xyz_Ri_binned = xyz.groupby_bins("Ri", Ri_bins, labels=Ri_bin_labels).mean()

fig, ax = plt.subplots(ncols=1, nrows=1, constrained_layout=True, sharey=True, figsize=(6, 5))

ax.scatter(xyz_Ri_binned.Ri, xyz_Ri_binned["ν_scaled"], color="black")

Ri = np.linspace(Ri_left, Ri_right, 200)
ν_max = float(xyz_Ri_binned["ν_scaled"].where(xyz_Ri_binned.Ri >= 0).max()) # 50e-4 # m²/s
ν0 = 50e-4 # m²/s
print(f"νₘₐₓ = {ν_max} for simulation {simname}; ν₀ = {ν0}")
Ri0_list = [0.8, 1.5, 3]
for Ri0 in Ri0_list:
    νt = ν0 * (1 - (Ri/Ri0)**2)**3
    νt[Ri<0] = ν0
    ax.plot(Ri, νt, label=f"$\\nu = \\nu_0 (1 - (Ri/Ri_0)^2)^3; \quad Ri_0 =$ {Ri0}")
ax.legend(); ax.grid(True)
ax.set_xlabel("Richardson number")
ax.set_ylabel("Viscosity [m²/s]")
ax.set_xlim(Ri_left, Ri_right)
ax.set_ylim(0, 1.15*max([ν_max, ν0]))

fig.savefig(f"figures/rev1_ν_vs_Ri_{simname}.pdf")
