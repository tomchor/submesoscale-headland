using Parameters


""" Rounds `a` to the nearest even number """
nearest_even(a) = Int(2*round(round(a) / 2))


""" Returns Nx Ny Nz based on total number of points `N`"""
function get_sizes(N; Lx=8, Ly=8, Lz=0.5, aspect_ratio_x=3.2, aspect_ratio_y=3.2)
    Cz = Lx * Ly / (Lz^2 * aspect_ratio_x * aspect_ratio_y)
    Nz = (N / Cz)^(1/3)
    Nx = Lx * Nz / (Lz * aspect_ratio_x)
    Ny = Ly * Nz / (Lz * aspect_ratio_y)
    Nx = round(Int, Nx)
    Ny = round(Int, Ny)
    Nz = round(Int, Nz)
    return (; Nx, Ny, Nz)
end


@with_kw struct Headland

    defaults = (;
                N = 85e6,

                V∞ = 0.01meters/second, # m/s
                H  = 40meters,

                α = 0.2, # Headland slope
                Fr_h = 0.1,
                Ro_h = 0.1,

                β = 1, # headland_intrusion_size / headland_width = "bumpiness" of feature (β=1 for an x-y symmetric seamount, e.g.)
                Lx_ratio = 3, # Lx / headland_intrusion_size_max
                Ly_ratio = 15, # Ly / headland_width

                Rz = 2.5e-3,
                runway_length_fraction_L = 4, # runway_length / L
                sponge_fraction_Ly = 0.1,
                T_advective_spinup = 20, # Should be a multiple of 20
                T_advective_statistics = 60, # Should be a multiple of 20
                )

    TEST = (; defaults...,)

    #+++ Ro=0.08
    R008F008 = (; defaults...,
                Ro_h = 0.08,
                Fr_h = 0.08,
                )

    R008F02 = (; defaults...,
                Ro_h = 0.08,
                Fr_h = 0.2,
                )

    R008F05 = (; defaults...,
                Ro_h = 0.08,
                Fr_h = 0.5,
                )

    R008F1 = (; defaults...,
                Ro_h = 0.08,
                Fr_h = 1.25,
                )
    #---

    #+++ Ro=0.2
    R02F008 = (; defaults...,
                Ro_h = 0.2,
                Fr_h = 0.08,
                )

    R02F02 = (; defaults...,
                Ro_h = 0.2,
                Fr_h = 0.2,
                )

    R02F05 = (; defaults...,
              Ro_h = 0.2,
              Fr_h = 0.5,
              )

    R02F1 = (; defaults...,
             Ro_h = 0.2,
             Fr_h = 1.25,
             )
    #---

    #+++ Ro=0.5
    R05F008 = (; defaults...,
               Ro_h = 0.5,
               Fr_h = 0.08,
               )

    R05F02 = (; defaults...,
              Ro_h = 0.5,
              Fr_h = 0.2,
              )

    R05F05 = (; defaults...,
              Ro_h = 0.5,
              Fr_h = 0.5,
              )

    R05F1 = (; defaults...,
             Ro_h = 0.5,
             Fr_h = 1.25,
             )
    #---

    #+++ Ro=1
    R1F008 = (; defaults...,
              Ro_h = 1.25,
              Fr_h = 0.08,
              )

    R1F02 = (; defaults...,
              Ro_h = 1.25,
              Fr_h = 0.2,
             )
 
    R1F05 = (; defaults...,
              Ro_h = 1.25,
              Fr_h = 0.5,
             )
 
    R1F1 = (; defaults...,
            Ro_h = 1.25,
            Fr_h = 1.25,
            )
    #---

    #+++ Gula: Ro=2
    R2F02 = (; defaults...,
              Ro_h = 2.0,
              Fr_h = 0.2,
              α = 0.1,
             )
     #---

end



function expand_headland_parameters(params)

    #+++ Unpack base parameters
    Ro_h = params.Ro_h
    Fr_h = params.Fr_h
    V∞ = V_inf = params.V∞
    α = params.α
    θ_rad = atan(α)
    #---

    #+++ Geometry
    Lz = 1.05* 2 * params.H # Lz is 5% larger than 2H in order to avoid flow constriction
    headland_intrusion_size_max = 2*params.H / α
    L = headland_intrusion_size_max / 2 # Horizontal length scale of the headland

    Lx = params.Lx_ratio * headland_intrusion_size_max
    Ly = params.Ly_ratio * L

    sponge_length_y = params.sponge_fraction_Ly * Ly
    runway_length = params.runway_length_fraction_L * L
    y_offset = sponge_length_y + runway_length
    #---

    #+++ Dynamically-relevant secondary parameters
    f₀ = f_0 = V∞ / (Ro_h * L)
    N²∞ = N2_inf = (V∞ / (Fr_h * params.H))^2
    R1 = √N²∞ * params.H / f₀
    z₀ = z_0 = params.Rz * params.H
    sponge_rate = 40 * V∞ / sponge_length_y
    #---

    #+++ Diagnostic parameters
    Γ = α * Fr_h # nonhydrostatic parameter (Schar 2002)
    Bu_h = (Ro_h / Fr_h)^2
    Slope_Bu = Ro_h / Fr_h # approximate slope Burger number
    @assert Slope_Bu ≈ α * √N²∞ / f₀
    #---

    #+++ Time scales
    T_inertial = 2π / f₀
    T_strouhal = L / (V∞ * 0.2)
    T_cycle = Ly / V∞
    T_advective = L / V∞
    #---

    #+++ Double check Ro_h and Fr_h
    Ro_h = V∞ / (f₀   * L)
    Fr_h = V∞ / (√N²∞ * params.H)
    @assert Ro_h ≈ params.Ro_h "Rossby number calculation doesn't match"
    @assert Fr_h ≈ params.Fr_h "Froude number calculation doesn't match"
    #---

    #+++ Get everything we just calculated we merge into `params`
    newparams = Base.@locals
    delete!(newparams, :params)
    return merge(params, NamedTuple(newparams))
    #---
end
