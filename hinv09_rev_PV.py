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

fig, axes = plt.subplots(nrows=3, constrained_layout=True, sharey=True, figsize=(10, 10))
yzsel = dict(yC=slice(600, 800), zC=slice(20, 60))
iyz = iyz.sel(**yzsel)
iyz.PV.pnplot(ax=axes[0], robust=True)
iyz.u.pnplot(ax=axes[1], robust=True)
iyz.Ri.pnplot(ax=axes[2], vmin=-2, vmax=2, cmap=BuRd)
for ax in axes:
    ax.axhline(y=z0, color="k", linestyle="--")
    iyz.b.pncontour(ax=ax, y="z", levels=100, colors="green", linewidths=1,)

fig, axes = plt.subplots(nrows=3, constrained_layout=True, sharey=True, figsize=(10, 10))
xzsel = dict(x=slice(-50, 350), z=slice(20, 60))
xiz.PV.pnsel(**xzsel).pnplot(ax=axes[0], robust=True)
xiz.u.pnsel(**xzsel).pnplot(ax=axes[1], robust=True)
xiz.Ri.pnsel(**xzsel).pnplot(ax=axes[2], vmin=-2, vmax=2, cmap=BuRd)
for ax in axes:
    ax.axhline(y=z0, color="k", linestyle="--")
    xiz.b.pncontour(ax=ax, y="z", levels=30, colors="y", linewidths=1,)

fig, axes = plt.subplots(ncols=3, constrained_layout=True, sharey=True, figsize=(10, 5))
xysel = dict(x=slice(-50, 350), y=slice(500, 1200))
xyi["1 + Ro"].pnsel(**xysel).pnplot(ax=axes[0], robust=True)
xyi.u.pnsel(**xysel).pnplot(ax=axes[1], robust=True)
xyi.Ri.pnsel(**xysel).pnplot(ax=axes[2], vmin=-2, vmax=2, cmap=BuRd)
for ax in axes:
    ax.axhline(y=y0, color="k", linestyle="--")

