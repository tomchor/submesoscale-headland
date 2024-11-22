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
xyz_N.yC.attrs = dict(long_name="$y$", units="m")
xyz_N.zC.attrs = dict(long_name="$z$", units="m")
xyz_N.v.attrs = dict(long_name="$v$", units="m/s")
xyz_N["∂u∂z"].attrs = dict(long_name="$\partial u/ \partial z$", units="1/s")

xyz_S = open_simulation("headland_simulations/data/xyz.NPN-R1F008-S.nc", decode_times=False, get_grid=False, unique=True, squeeze=True, use_advective_periods=True)
xyz_S = xyz_S.sel(time=30, method="nearest")
xyz_S.yC.attrs = dict(long_name="$y$", units="m")
xyz_S.zC.attrs = dict(long_name="$z$", units="m")
xyz_S.v.attrs = dict(long_name="$v$", units="m/s")
xyz_S["∂u∂z"].attrs = dict(long_name="$\partial u/ \partial z$", units="1/s")

fig, axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True, sharey=True, figsize = (8, 4.5), sharex=True, )

cbar_kwargs = dict(shrink=0.8, pad=0.01, aspect=30, location="bottom")
xyz_N["v"].pnsel(x=100, method="nearest").pnsel(y=slice(-200, 1800)).pnplot(ax=axes[0, 0], vmin=-1e-2, vmax=+1e-2, cmap=BuRd, rasterized=True, add_colorbar=False)
xyz_S["v"].pnsel(x=100, method="nearest").pnsel(y=slice(-200, 1800)).pnplot(ax=axes[1, 0], vmin=-1e-2, vmax=+1e-2, cmap=BuRd, rasterized=True, cbar_kwargs=cbar_kwargs)

xyz_N["∂u∂z"].pnsel(x=100, method="nearest").pnsel(y=slice(-200, 1800)).pnplot(ax=axes[0, 1], vmin=-3e-3, vmax=+3e-3, cmap=BuRd, rasterized=True, add_colorbar=False)
xyz_S["∂u∂z"].pnsel(x=100, method="nearest").pnsel(y=slice(-200, 1800)).pnplot(ax=axes[1, 1], vmin=-3e-3, vmax=+3e-3, cmap=BuRd, rasterized=True, cbar_kwargs=cbar_kwargs)

for ax in axes.flatten():
    ax.set_title("")
axes[0, 0].set_xlabel("")
axes[0, 1].set_xlabel("")
axes[0, 1].set_ylabel("")
axes[1, 1].set_ylabel("")

letterize(axes.flatten(), x=0.05, y=0.9, fontsize=14)

fig.savefig(f"figures/cyclonic_coupled.pdf")
