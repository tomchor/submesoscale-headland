import xarray as xr
import numpy as np
from matplotlib import pyplot as plt

α_array = np.array([0.07, 0.1, 0.2])
V_array = np.array([0.5, 1]) # m
N_array = np.array([7e-3, 10e-3]) # 1/s

α = xr.DataArray(α_array, dims=["α"], coords=dict(α=α_array))
H = 400 # m; half the total depth, which is ≈ 800 m
L = H / α # m; half the intrusion length at the bottom (approx between 4 and 8 km for the bahamas)
V = xr.DataArray(V_array, dims=["V"], coords=dict(V=V_array))
N = xr.DataArray(N_array, dims=["N"], coords=dict(N=N_array))
f = 6.6e-5 # 1/s

Ro_h = V / (f * L)
Fr_h = V / (N * H)
S_h  = Ro_h / Fr_h

ΔLa = 28 - 25.5
ΔLo = 79.5 - 78

Δy = ΔLa * 111e3 # m
Δx = ΔLo * 100e3 # m
Δz = 800 # m
ΔV = Δx * Δy * Δz # m³

#Dissip_gula_W = 0.5e9 # W
#Dissip_gula_m6s3
#m2/s3
ρ0 = 1000
# 1 Watt = kg m² / s³
