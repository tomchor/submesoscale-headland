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
xyz = xyz.sel(time=30)

xyz["1 + Ro"] = 1 + xyz.Ro

x0 = 200
y0 = 700
z0 = 40
iyz = xyz.sel(xC=x0, method="nearest")
xiz = xyz.sel(yC=y0, method="nearest")
xyi = xyz.sel(zC=z0, method="nearest")

#fig, axes = plt.subplots(nrows=3, constrained_layout=True, sharey=True, figsize=(10, 10))
#yzsel = dict(yC=slice(600, 800), zC=slice(30, 50))
#iyz = iyz.sel(**yzsel)
#iyz.PV.pnplot(ax=axes[0], robust=True)
#iyz.ω_x.pnplot(ax=axes[1], robust=True)
#iyz.Ri.pnplot(ax=axes[2], vmin=-2, vmax=2, cmap=BuRd)
#for ax in axes:
#    ax.axhline(y=z0, color="k", linestyle="--")
#    iyz.b.pncontour(ax=ax, y="z", levels=50, colors="green", linewidths=1,)
#pause

fig, axes = plt.subplots(nrows=2, constrained_layout=True, sharex=True, sharey=True, figsize=(10, 8))
xzsel = dict(xC=slice(0, 255), zC=slice(30, 50))
xiz = xiz.sel(**xzsel)
xiz.PV.pnplot(ax=axes[0], robust=True, rasterized=True)
xiz.Ri.pnplot(ax=axes[1], vmin=-2, vmax=2, cmap=BuRd, rasterized=True)
for ax in axes:
    ax.axhline(y=z0, color="k", linestyle="--")
CS = xiz.b.pncontour(ax=axes[-1], y="z", levels=np.linspace(0.000396, 0.00041, 6), colors="y", linewidths=1,)
axes[1].clabel(CS, CS.levels, inline=True, fontsize=8)
fig.savefig(f"figures_check/hinv_PV_xz.pdf")

fig, axes = plt.subplots(ncols=2, constrained_layout=True, sharey=True, figsize=(10, 7))
xysel = dict(x=slice(-50, 350), y=slice(500, 1200))
xyi.PV.pnsel(**xysel).pnplot(ax=axes[0], robust=True, rasterized=True)
xyi.u.pnsel(**xysel).pnplot(ax=axes[1], robust=True, rasterized=True)
#xyi.Ri.pnsel(**xysel).pnplot(ax=axes[2], vmin=-2, vmax=2, cmap=BuRd, rasterized=True)
for ax in axes:
    ax.axhline(y=y0, color="k", linestyle="--")
fig.savefig(f"figures_check/hinv_PV_xy.pdf")

