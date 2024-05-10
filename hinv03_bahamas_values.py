import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
g = 9.81 # m/s²
ρ0 = 1000 # kg/m³

#+++ Gula et al.'s results
α_array_Gula = np.array([0.07, 0.1, 0.2])
V_array_Gula = np.array([0.5, 1]) # m
N_array_Gula = np.array([7e-3, 10e-3]) # 1/s

α_Gula = xr.DataArray(α_array_Gula, dims=["α"], coords=dict(α=α_array_Gula))
H_Gula = 400 # m; half the total depth, which is ≈ 800 m
L_Gula = H_Gula / α_Gula # m; half the intrusion length at the bottom (approx between 4 and 8 km for the bahamas)
V_Gula = xr.DataArray(V_array_Gula, dims=["V"], coords=dict(V=V_array_Gula))
N_Gula = xr.DataArray(N_array_Gula, dims=["N"], coords=dict(N=N_array_Gula))
f_Gula = 6.6e-5 # 1/s

Ro_h = V_Gula / (f_Gula * L_Gula)
Fr_h = V_Gula / (N_Gula * H_Gula)
Sb_h  = Ro_h / Fr_h

ΔLa = 28 - 25.5
ΔLo = 79.5 - 78

Δx_Gula = ΔLo * 100e3 # m
Δy_Gula = ΔLa * 111e3 # m
Δz_Gula = 800 # m
ΔV_Gula = Δx_Gula * Δy_Gula * Δz_Gula # m³

ε_scale_Gula = V_Gula**3 / L_Gula

norm = xr.Dataset()

norm["εₖ_max_Gula"] = 1e-5
norm["εₖ_max_norm_Gula"] = norm["εₖ_max_Gula"] / ε_scale_Gula

# Reminder: 1 Watt = kg m² / s³
norm["ρ∫∫∫ε̄ₖdxdydz_Gula"] = 0.5e9 # W
ρ = 1000
norm["∫∫∫ε̄ₖdxdydz_Gula"] = norm["ρ∫∫∫ε̄ₖdxdydz_Gula"] / ρ
norm["⟨ε̄ₖ⟩_Gula"] = norm["∫∫∫ε̄ₖdxdydz_Gula"] / ΔV_Gula
norm["εₖ_norm_Gula"] = norm["⟨ε̄ₖ⟩_Gula"] / ε_scale_Gula
#---

#+++ Nagai et alia's results
α_array_Nagai = np.array([0.05, 0.1, 0.2])
V_array_Nagai = np.array([0.5, 1]) # m

Δρ2 = 25.5 - 24.5 # kg/m³
Δh2 = 100 # m
Δρ1 = 26.0 - 24.5 # kg/m³
Δh1 = 500 # m
N1 = np.sqrt((g / ρ0) * Δρ1 / Δh1)
N2 = np.sqrt((g / ρ0) * Δρ2 / Δh2)
N_array_Nagai = np.array([N1, N2]) # 1/s

α_Nagai = xr.DataArray(α_array_Nagai, dims=["α"], coords=dict(α=α_array_Nagai))
H_Nagai = 200 # m; half the total depth, which is ≈ 400 m
L_Nagai = H_Nagai / α_Nagai # m; half the intrusion length at the bottom (approx between 2 and 5 km for the kuroshio islands)
V_Nagai = xr.DataArray(V_array_Nagai, dims=["V"], coords=dict(V=V_array_Nagai))
N_Nagai = xr.DataArray(N_array_Nagai, dims=["N"], coords=dict(N=N_array_Nagai))
f_Nagai = 7.3e-5 # 1/s

Ro_h = V_Nagai / (f_Nagai * L_Nagai)
Fr_h = V_Nagai / (N_Nagai * H_Nagai)
Sb_h  = Ro_h / Fr_h
pause
#---

#+++ Chor and Wenegrat's results
Δx_Chor = (800 + 400) # m
Δy_Chor = 3000 # m
Δz_Chor = 84 # m
ΔV_Chor = Δx_Chor * Δy_Chor * Δz_Chor

ε_scale_Chor = 0.01**3 / 400 # (m/s)³ / m

norm["εₖ_max_Chor"] = 1e-8
norm["εₖ_max_norm_Chor"] = norm["εₖ_max_Chor"] / ε_scale_Chor

norm["∫∫∫ε̄ₖdxdydz_Chor"] = 1e-2 # W m³ / kg
norm["ρ∫∫∫ε̄ₖdxdydz_Chor"] = ρ * norm["∫∫∫ε̄ₖdxdydz_Chor"]
norm["⟨ε̄ₖ⟩_Chor"] = norm["∫∫∫ε̄ₖdxdydz_Chor"] / ΔV_Chor
norm["εₖ_norm_Chor"] = norm["⟨ε̄ₖ⟩_Chor"] / ε_scale_Chor
#---

#+++ Print results
α_value = 0.07
N_value = 0.01
norm = norm.sel(α=α_value)
Ro_h = Ro_h.sel(α=α_value)
Fr_h = Fr_h.sel(N=N_value)
Sb_h = Sb_h.sel(α=α_value, N=N_value)
for V in norm.V.values:
    print(f"\n---- For V∞ = {V} m/s ----\n")
    print(f"Gula's Roₕ = ", Ro_h.sel(V=V).item())
    print(f"Gula's Frₕ = ", Fr_h.sel(V=V).item())
    print(f"Gula's Sbₕ = ", Sb_h.sel(V=V).item())
    print()

    print("Gula et al.'s normalized instantaneous dissipation: ", norm["εₖ_max_norm_Gula"].sel(V=V).item())
    print("Chor & Wenegrat's normalized instantaneous dissipation: ", norm["εₖ_max_norm_Chor"].item())
    print()
    print("Gula et al.'s normalized average dissipation: ", norm["εₖ_norm_Gula"].sel(V=V).item(), "m²/s³")
    print("Chor & Wenegrat's normalized average dissipation: ", norm["εₖ_norm_Chor"].item(), "m²/s³")
    print()
