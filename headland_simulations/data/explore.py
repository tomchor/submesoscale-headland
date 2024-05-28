import xarray as xr
from matplotlib import pyplot as plt

tti = xr.load_dataset("tti.NPN-R008F008-f4.nc", decode_times=False).squeeze()
tti["uᵢ∂ᵢp"].plot(x="xC", col="time", col_wrap=4, robust=True)

xyi = xr.open_dataset("xyi.NPN-R008F008-f4.nc", decode_times=False).squeeze()
xyi = xyi[["uᵢ∂ᵢp"]].load()
xyi["uᵢ∂ᵢp"].isel(time=slice(None, None, 22)).plot(x="xC", col="time", col_wrap=4, robust=True)

plt.figure()
xyi["uᵢ∂ᵢp"].std(("xC", "yC")).plot(yscale="log")

