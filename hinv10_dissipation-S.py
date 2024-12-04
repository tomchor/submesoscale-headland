import numpy as np
import pynanigans as pn
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from cmocean import cm
from aux02_plotting import BuRd, letterize
from aux00_utils import open_simulation
plt.rcParams["figure.constrained_layout.use"] = True
plt.rcParams["font.size"] = 9

xyz_N = open_simulation("headland_simulations/data/xyz.NPN-R1F008.nc", decode_times=False, get_grid=False, unique=True, squeeze=True, use_advective_periods=True)
xyz_N = xyz_N.sel(time=30, method="nearest")

xyz_S = open_simulation("headland_simulations/data/xyz.NPN-R1F008-S.nc", decode_times=False, get_grid=False, unique=True, squeeze=True, use_advective_periods=True)
xyz_S = xyz_S.sel(time=30, method="nearest")


fig, axes = plt.subplots(ncols=1, nrows=4, constrained_layout=True, sharey=True, figsize=(8, 8))
xyz_N.ω_y.pnsel(y=700, method="nearest").pnsel(x=slice(-100, None)).pnplot(ax=axes[0], robust=True)
xyz_N.PV.pnsel (y=700, method="nearest").pnsel(x=slice(-100, None)).pnplot(ax=axes[1], robust=True)
xyz_N.Ri.pnsel (y=700, method="nearest").pnsel(x=slice(-100, None)).pnplot(ax=axes[2], vmin=-2, vmax=2, cmap=BuRd)
xyz_N["εₖ"].pnsel (y=700, method="nearest").pnsel(x=slice(-100, None)).pnplot(ax=axes[3], vmin=1e-11, vmax=1e-9, norm=LogNorm(clip=True))

pause


xyi_N = open_simulation("headland_simulations/data/xyi.NPN-R1F008.nc", decode_times=False, get_grid=False, unique=True, squeeze=True, use_advective_periods=True)
xyi_N = xyi_N.sel(time=30, method="nearest")
xyi_N["q"] = xyi_N.PV / (xyi_N.attrs["N²∞"] * xyi_N.attrs["f₀"])

xyi_S = open_simulation("headland_simulations/data/xyi.NPN-R1F008-S.nc", decode_times=False, get_grid=False, unique=True, squeeze=True, use_advective_periods=True)
xyi_S = xyi_S.sel(time=30, method="nearest")
xyi_S["q"] = xyi_S.PV / (xyi_S.attrs["N²∞"] * xyi_S.attrs["f₀"])

fig, axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True, sharey=True, figsize=(8, 8))

xsel = dict(x=slice(-300, np.inf), y=slice(0, np.inf))
xyi_N.q.pnsel(**xsel).pnplot(ax=axes[0,0], vmin=-7.5, vmax=+7.5, cmap=BuRd)
xyi_S.q.pnsel(**xsel).pnplot(ax=axes[1,0], vmin=-7.5, vmax=+7.5, cmap=BuRd)

xyi_N["εₖ"].pnsel(**xsel).pnplot(ax=axes[0,1], vmin=1e-11, vmax=1e-9, norm=LogNorm(clip=True))
xyi_S["εₖ"].pnsel(**xsel).pnplot(ax=axes[1,1], vmin=1e-11, vmax=1e-9, norm=LogNorm(clip=True))

