import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm

ds = xr.load_dataset("test_pressure.nc", decode_times=False).squeeze()

ds.pNHS.std(("xC", "yC", "zC")).plot()

ds.pNHS.isel(zC=0, time=slice(None, None, 11)).plot(x="xC", col="time", col_wrap=5, robust=True, norm=SymLogNorm(linthresh=1e-3))
for ax in plt.gcf().axes:
    ax.axvline(x=400)

