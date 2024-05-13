import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
g = 9.81 # m/s²
ρ0 = 1000 # kg/m³
# Reminder: 1 Watt = kg m² / s³

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

Gula = xr.Dataset()

Gula["Ro_h"] = V_Gula / (f_Gula * L_Gula)
Gula["Fr_h"] = V_Gula / (N_Gula * H_Gula)
Gula["Sb_h"] = Gula.Ro_h / Gula.Fr_h

ΔLa = 28 - 25.5
ΔLo = 79.5 - 78

Δx_Gula = ΔLo * 100e3 # m
Δy_Gula = ΔLa * 111e3 # m
Δz_Gula = 800 # m
ΔV_Gula = Δx_Gula * Δy_Gula * Δz_Gula # m³

Gula["ε_scale"] = V_Gula**3 / L_Gula

Gula["εₖ_max"] = 1e-5
Gula["εₖ_max_norm"] = Gula["εₖ_max"] / Gula.ε_scale

Gula["ρ∫∫∫ε̄ₖdxdydz"] = 0.5e9 # W
ρ = 1000
Gula["∫∫∫ε̄ₖdxdydz"] = Gula["ρ∫∫∫ε̄ₖdxdydz"] / ρ
Gula["⟨ε̄ₖ⟩"] = Gula["∫∫∫ε̄ₖdxdydz"] / ΔV_Gula
Gula["εₖ_norm"] = Gula["⟨ε̄ₖ⟩"] / Gula.ε_scale
#---

#+++ Nagai et alia's results
α_array_Nagai = np.array([0.05, 0.1, 0.2])
V_array_Nagai = np.array([0.5, 1]) # m

Δρ1 = 26.0 - 24.5 # kg/m³
Δh1 = 500 # m
Δρ2 = 25.5 - 24.5 # kg/m³
Δh2 = 100 # m
N1 = np.sqrt((g / ρ0) * Δρ1 / Δh1)
N2 = np.sqrt((g / ρ0) * Δρ2 / Δh2)
N_array_Nagai = np.array([N1, N2]) # 1/s

α_Nagai = xr.DataArray(α_array_Nagai, dims=["α"], coords=dict(α=α_array_Nagai))
H_Nagai = 200 # m; half the total depth, which is ≈ 400 m
L_Nagai = H_Nagai / α_Nagai # m; half the intrusion length at the bottom (approx between 2 and 5 km for the kuroshio islands)
V_Nagai = xr.DataArray(V_array_Nagai, dims=["V"], coords=dict(V=V_array_Nagai))
N_Nagai = xr.DataArray(N_array_Nagai, dims=["N"], coords=dict(N=N_array_Nagai))
f_Nagai = 7.3e-5 # 1/s

Nagai = xr.Dataset()

Nagai["Ro_h"] = V_Nagai / (f_Nagai * L_Nagai)
Nagai["Fr_h"] = V_Nagai / (N_Nagai * H_Nagai)
Nagai["Sb_h"] = Nagai.Ro_h / Nagai.Fr_h

Nagai["ε_scale"] = V_Nagai**3 / L_Nagai

Nagai["εₖ_max"] = 10**(-6.5)
Nagai["εₖ_max_norm"] = Nagai["εₖ_max"] / Nagai.ε_scale
#---

#+++ Chor and Wenegrat's results
Δx_Chor = (800 + 400) # m
Δy_Chor = 3000 # m
Δz_Chor = 84 # m
ΔV_Chor = Δx_Chor * Δy_Chor * Δz_Chor

L_Chor = 200 # m
V_Chor = 0.01 # m

Chor = xr.Dataset()

Chor["ε_scale"] = V_Chor**3 / L_Chor # (m/s)³ / m

Chor["εₖ_max"] = 1e-8
Chor["εₖ_max_norm"] = Chor["εₖ_max"] / Chor.ε_scale

Chor["∫∫∫ε̄ₖdxdydz"] = 1e-2 # W m³ / kg
Chor["ρ∫∫∫ε̄ₖdxdydz"] = ρ * Chor["∫∫∫ε̄ₖdxdydz"]
Chor["⟨ε̄ₖ⟩"] = Chor["∫∫∫ε̄ₖdxdydz"] / ΔV_Chor
Chor["εₖ_norm"] = Chor["⟨ε̄ₖ⟩"] / Chor.ε_scale
#---

#+++ Print results
N_value = 0.01
α_value_Gula = 0.07
Gula = Gula.sel(α=α_value_Gula, N=N_value)

α_value_Nagai = 0.2
Nagai = Nagai.sel(α=α_value_Nagai, N=N_value, method="nearest")

for V in Gula.V.values:
    print(f"\n---- For V∞ = {V} m/s ----\n")
    print(f"Gula's Roₕ = ", Gula.Ro_h.sel(V=V).item())
    print(f"Gula's Frₕ = ", Gula.Fr_h.sel(V=V).item())
    print(f"Gula's Sbₕ = ", Gula.Sb_h.sel(V=V).item())
    print()
    print(f"Nagai's Roₕ = ", Nagai.Ro_h.sel(V=V).item())
    print(f"Nagai's Frₕ = ", Nagai.Fr_h.sel(V=V).item())
    print(f"Nagai's Sbₕ = ", Nagai.Sb_h.sel(V=V).item())
    print()

    print("Gula et al.'s normalized instantaneous dissipation: ", Gula["εₖ_max_norm"].sel(V=V).item())
    print("Chor & Wenegrat's normalized instantaneous dissipation: ", Chor["εₖ_max_norm"].item())
    print()
    print("Gula et al.'s normalized average dissipation: ", Gula["εₖ_norm"].sel(V=V).item(), "m²/s³")
    print("Chor & Wenegrat's normalized average dissipation: ", Chor["εₖ_norm"].item(), "m²/s³")
    print()
