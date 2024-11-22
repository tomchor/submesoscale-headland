import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from cmocean import cm
from aux02_plotting import BuRd, letterize
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

xyi = xr.open_dataset("headland_simulations/data/xyi.NPN-R1F008.nc", decode_times=False).squeeze()
xyi = xyi.assign_coords(time=xyi.time/xyi.T_advective)

xy0 = xyi.sel(time=30)
xy0["1 + Ro"] = 1 + xy0.Ro

fig, axes = plt.subplots(ncols=3, constrained_layout=True, sharey=True, figsize=(10, 5))

xsel = dict(x=slice(-50, 350), y=slice(500, 1200))
xy0["1 + Ro"].pnsel(**xsel).pnplot(ax=axes[0], robust=True)
xy0.u.pnsel(**xsel).pnplot(ax=axes[1], robust=True)
xy0.Ri.pnsel(**xsel).pnplot(ax=axes[2], vmin=-2, vmax=2, cmap=BuRd)

