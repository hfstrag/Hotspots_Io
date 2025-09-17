#### Simulation des hotspots volcaniques d’Io
## Fonctions utilitaires

# Dépendances
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as C
import pandas as pd

# ----- constantes fondamentales ------
h = C.h
c = C.c
kB = C.k
sigma = C.Stefan_Boltzmann

# ----- fonctions utiles -----
#fonction de Planck
def planck_lambda_per_um(wl_um, T):
    """
    Radiance spectrale de Planck par µm.
    wl_um : scalaire ou array (µm)
    T     : scalaire ou array (K)
    Retourne W·m^-2·sr^-1·µm^-1
    """
    wl_um = np.atleast_1d(wl_um)
    T = np.atleast_1d(T)

    wl_m = wl_um * 1e-6  # µm -> m
    x = (C.h * C.c) / (wl_m[:, None] * C.k * T[None, :])  # broadcasting
    B_m = (2 * C.h * C.c**2) / (wl_m[:, None]**5 * np.expm1(x))  # W·m^-2·sr^-1·m^-1
    B_um = B_m * 1e-6  # conversion en µm^-1

    if B_um.shape[1] == 1:
        return B_um[:, 0]
    return B_um

#fonction pour obtenir la radiance de fond (reflectance)
def get_radiance(wvl_um, i=0.0, e=0.0, filename=None):
    """
    Charge un spectre de radiance depuis un CSV et l’interpole sur une grille en µm.
    CSV attendu : colonnes [Wavelength (nm), Radiance (W/m²/sr/µm)]
    """
    data = pd.read_csv(filename, header=0)
    data = data.rename(columns={
        "Wavelength (nm)": "wvl_nm",
        "Radiance (W/m²/sr/µm)": "rad"
    })
    wvl_file_um = data["wvl_nm"].to_numpy() / 1000.0   # nm -> µm
    rad_file = data["rad"].to_numpy()

    rad_interp = np.interp(wvl_um, wvl_file_um, rad_file, left=0, right=0)

    mu_i = np.cos(np.radians(i))
    mu_e = np.cos(np.radians(e))
    return rad_interp * mu_i * mu_e

# ----- seuil NIMS -----
NIMS_wl_um = 4.7      # µm
NIMS_threshold = planck_lambda_per_um(np.array([NIMS_wl_um]), 180.0)[0]  # radiance seuil (180 K)


# ----- fonction de simulation pour un seul hotspot de rayon r0----
def simulate_one(Tmax, r0_km, periode="jour", Tmin=130.0, d_MAJIS=57e3,
                 wl_um=np.linspace(0.7016, 5.018, 300),
                 r_infty_factor=8, nr_radial=2000,
                 use_background_files=True):
    r0 = r0_km * 1e3
    nr = nr_radial
    r = np.linspace(0.0, r_infty_factor * r0, nr)
    dr = r[1] - r[0]
    dA_ring = 4.0 * np.pi * r * dr
    norm = d_MAJIS**2

    T_r = (Tmax - Tmin) * np.exp(- (r**2) / (r0**2)) + Tmin
    B_wl_r = planck_lambda_per_um(wl_um, T_r)
    numer = np.sum(B_wl_r * dA_ring, axis=1)
    L_hotspot_avg = numer / norm

    if periode == "jour" and use_background_files:
        io_rad_nims = get_radiance(wl_um, i=14.13, e=0, filename="Io_NIMS.csv")
        io_rad_jiram = get_radiance(wl_um, i=14.13, e=0, filename="Io_JIRAM_JM0261.csv")
        mask = (wl_um > 2.5) & (wl_um < 5.0)
        norm_factor = np.mean(io_rad_jiram[mask]) / np.mean(io_rad_nims[mask])
        L_background = io_rad_nims * norm_factor
    else:
        L_background = planck_lambda_per_um(wl_um, Tmin)

    f_fill = min((np.pi * r0**2) / (d_MAJIS**2), 1.0)
    L_pixel = L_background + L_hotspot_avg

    integrated_radiance_sr = np.trapezoid(L_pixel, wl_um)
    exitance = np.pi * integrated_radiance_sr
    T_eff = (exitance / sigma) ** 0.25

    P_tot = 2 * np.pi * np.sum(sigma * T_r**4 * r * dr)

    r_eq = np.sqrt(P_tot / (sigma * T_eff**4 * np.pi))
    # --- détectabilité NIMS ---
    # radiance du pixel à 4.7 µm
    idx_47 = np.argmin(np.abs(wl_um - NIMS_wl_um))
    rad_47 = L_pixel[idx_47]

    detectable = rad_47 >= NIMS_threshold

    return {
        "r0_km": r0_km,
        "L_pixel": L_pixel,
        "T_eff": float(T_eff),
        "f_fill": f_fill,
        "power_W": P_tot,
        "r_eq": r_eq,       
    }


# ----- fonction pour balayer une liste de r0 -----
def run_sweep(Tmax, r0_list_km, periode="jour", Tmin=130.0, d_MAJIS=57e3,
              wl_um=np.linspace(0.7016, 5.018, 300),
              r_infty_factor=8, nr_radial=2000,
              use_background_files=True):
    results = []
    detectable_list = []
    for r0 in r0_list_km:
        results.append(simulate_one(Tmax, r0, periode, Tmin, d_MAJIS,
                                    wl_um, r_infty_factor, nr_radial,
                                    use_background_files))
    return results


# ----- fonction pour estimer r0 à partir d’observations -----
def estimate_r0_from_obs(Teff_obs, P_obs_GW, Tmax, r0_grid_km,
                         periode="jour", Tmin=130.0, d_MAJIS=57e3,
                         wl_um=np.linspace(0.7016,5.018,300),
                         r_infty_factor=8, nr_radial=1000,
                         sigma_T=10.0, sigma_logP=0.2,
                         use_background_files=True, verbose=False):
    best = None
    diagnostics = []
    P_obs_W = P_obs_GW * 1e9

    for r0_km in r0_grid_km:
        res = simulate_one(Tmax=Tmax, r0_km=r0_km, periode=periode, Tmin=Tmin,
                           d_MAJIS=d_MAJIS, wl_um=wl_um,
                           r_infty_factor=r_infty_factor, nr_radial=nr_radial,
                           use_background_files=use_background_files)
        Tmod = res["T_eff"]
        Pmod = res["power_W"]
        if Pmod <= 0 or P_obs_W <= 0:
            chi2 = np.inf
        else:
            chi2 = ((Tmod - Teff_obs)/sigma_T)**2 + ((np.log10(Pmod) - np.log10(P_obs_W))/sigma_logP)**2

        diagnostics.append({"r0_km": r0_km, "Tmod": Tmod,
                            "Pmod_GW": Pmod/1e9, "chi2": chi2,
                            "f_fill": res["f_fill"]})
        if best is None or chi2 < best["chi2"]:
            best = {"r0_km": r0_km, "chi2": chi2,
                    "Tmod": Tmod, "Pmod_GW": Pmod/1e9,
                    "f_fill": res["f_fill"], "res": res}

    if verbose:
        print("Diagnostics (first 10):")
        for d in diagnostics[:10]:
            print(d)

    return best, diagnostics
