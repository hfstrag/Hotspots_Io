
import Fonctions as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as C
import pandas as pd

if __name__ == "__main__":

    # paramètres de base
    periode = "jour"
    Tmax = 1600.0
    Tmin = 130.0
    d_MAJIS = 57e3
    r0_list_km = [0.1,0.2,0.4,0.8,1.6,3.2,6.4,12.8,25.6,51.2]
    wl_um = np.linspace(0.7016,5.018,300)

    # simulation
    sweep = F.run_sweep(Tmax, r0_list_km, periode=periode, Tmin=Tmin, d_MAJIS=d_MAJIS, wl_um=wl_um)

    # affichage résultats
    print("r0 (km)   T_eff (K)   P_tot (GW)   f_fill")
    for s in sweep:
        print(f"{s['r0_km']:6.2f}  {s['T_eff']:8.2f}  {s['power_W']/1e9:10.2f}  {s['f_fill']:6.3f}")

    # Spectres
    plt.figure(figsize=(8,5))
    for s in sweep:
        plt.semilogy(wl_um, s["L_pixel"], label=f"r0={s['r0_km']} km")
    plt.xlabel("Wavelength (µm)")
    plt.ylabel("Radiance (W·m$^{-2}$·sr$^{-1}$·µm$^{-1}$)")
    plt.title("Spectres simulés du pixel")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Spectres_simules.png", dpi=300)

    # Profils T(r)
    plt.figure(figsize=(8,5))
    for r0_km in r0_list_km:
        r_km = np.logspace(-3, np.log10(d_MAJIS/1000), 300)
        T_r = (Tmax - Tmin) * np.exp(- ((r_km*1e3)**2) / ((r0_km*1e3)**2)) + Tmin
        plt.plot(r_km, T_r, label=f"r0={r0_km} km")
    plt.xscale("log")
    plt.xlabel("Radius (km)")
    plt.ylabel("Temperature (K)")
    plt.title("Profils radiaux T(r)")
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.tight_layout()
    plt.savefig("Profils_Tr.png", dpi=300)

    # T_eff vs r0
    plt.figure(figsize=(7,5))
    plt.plot([s["r0_km"] for s in sweep], [s["T_eff"] for s in sweep], "o-", lw=2)
    plt.xscale("log")
    plt.xlabel("Hotspot radius r0 (km)")
    plt.ylabel("Effective temperature T_eff (K)")
    plt.title("T_eff en fonction de la taille du hotspot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("Teff_vs_r0.png", dpi=300)
    plt.show()


    # Exemple estimation à partir d’observations
    Teff_obs = 220.0  # K (exemple)
    P_obs_GW = 56.0   # GW (exemple)
    r0_grid_km = np.logspace(-2, 1.5, 40)

    best, diag = F.estimate_r0_from_obs(Teff_obs, P_obs_GW, Tmax, r0_grid_km, verbose=True)
    print("\n>>> Estimation r0 à partir des observations :")
    print("Best-fit r0 (km):", best["r0_km"])
    print("Model T_eff:", best["Tmod"], "K ; Model P (GW):", best["Pmod_GW"])
    print("Fill factor:", best["f_fill"], "chi2:", best["chi2"])



