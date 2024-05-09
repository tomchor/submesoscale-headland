import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm

ds = xr.load_dataset("xyz.NPN-TEST-f32.nc", decode_times=False).squeeze()

ds.pNHS.std(("xC", "yC", "zC")).plot()

ds.pNHS.isel(zC=0, time=slice(None, None, 50)).plot(x="xC", col="time", col_wrap=5, robust=True, norm=SymLogNorm(linthresh=1e-3))


