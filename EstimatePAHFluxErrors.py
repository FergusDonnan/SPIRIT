#!/usr/bin/env python3
"""
Residual-based PAH flux uncertainties for SPIRIT Quick fits.

For each individual Drude (Output.csv Name=='PAH'), each IrModel.py PAH
complex, and the synthetic PAH 3.4 feature (sum of Drudes at 3.395 + 3.405):

  1. W_eff = F / F_λ,peak
     F  = Strength (10^-17 W/m^2) from Output.csv
     F_λ,peak from summed PAH component profile(s), converted
     F_ν → F_λ as in IrModel.py:  F_λ = F_ν * 1e-9 * c / λ^2

  2. residual = data − full model  (Spectrum.txt vs FullModel.txt)
     then apply SetupFit.apply_emission_line_mask to the residual.

  3. Noise term (in the window [λ_peak − W_eff/2, λ_peak + W_eff/2]):
     σ_λ = std of masked residuals in F_λ units
     σ_noise = σ_λ * sqrt(Σ δλ_i^2)

  4. Continuum term (provisional):
     ΔC = δ_c * C   with δ_c = 0.02 (2% of local continuum from CSV)
     σ_cont = ΔC * W_eff

  5. σ_F = sqrt(σ_noise^2 + σ_cont^2),  n_σ = F / σ_F

Integrated into SPIRIT Quick fits via apply_residual_pah_errors_to_quick_output()
(called from IrModel.RunFit after Output.csv is written).

CLI:
  python EstimatePAHFluxErrors.py --summary-batch
      → Seyf nuclei + SPIRIT regions → PAH_FluxError_Tests/Seyf_SPIRIT_PAH_flux_errors_2pct.csv
  python EstimatePAHFluxErrors.py --object /path/to/obj [--no-plots]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import SetupFit

# Speed of light [µm Hz] as used in IrModel.py (nu = c / wav)
C_UM_HZ = 2.9979246e14
# IrModel converts model F_ν → CSV Strength / Continuum with this factor
FNU_TO_FLAMBDA_FACTOR = 1.0e-9
# Provisional fractional continuum uncertainty
DELTA_C_FRAC = 0.02

# Complex membership — rest wavelengths as in IrModel.py
PAH_COMPLEXES = {
    "PAH 3.3 Complex": (3.3, (3.29,)),
    "PAH 5.2 Complex": (5.237, (5.185, 5.237, 5.275)),
    "PAH 5.7 Complex": (5.7, (5.64, 5.7, 5.76)),
    "PAH 6.2 Complex": (6.2, (6.2,)),
    "PAH 7.7 Complex": (7.7, (7.42, 7.55, 7.61, 7.82)),
    "PAH 8.6 Complex": (8.6, (8.5, 8.61)),
    "PAH 11.3 Complex": (11.3, (11.20, 11.26, 11.25)),
    "PAH 12.7 Complex": (12.7, (12.6, 12.77)),
    "PAH 17.0 Complex": (17.0, (16.45, 17.04, 17.375)),
}

# Synthetic 3.4 µm aliphatic feature: sum of these two Drudes (not an IrModel complex)
PAH34_DRUDE_CENTS = (3.395, 3.405)
PAH34_NOMINAL_WAV = 3.4

# Short feature keys for the wide summary CSV
SUMMARY_FEATURES = (
    ("PAH3.3", "complex", "PAH 3.3 Complex"),
    ("PAH3.4", "pah34", None),
    ("PAH6.2", "complex", "PAH 6.2 Complex"),
    ("PAH7.7", "complex", "PAH 7.7 Complex"),
    ("PAH11.3", "complex", "PAH 11.3 Complex"),
    ("PAH12.7", "complex", "PAH 12.7 Complex"),
    ("PAH17.0", "complex", "PAH 17.0 Complex"),
)

PLOT_COMPLEXES = ("PAH 7.7 Complex", "PAH 11.3 Complex")

SEYF_NUCLEI_BASE = Path("/Users/frdonnan/Downloads/Seyf_Nuclei")
SPIRIT_RESULTS_BASE = Path(__file__).resolve().parent / "Results"

SEYF_NUCLEI_IDS = [
    "eso137",
    "eso420",
    "ic5063",
    "mcg05",
    "ngc3081",
    "ngc3227",
    "ngc4051",
    "ngc5506",
    "ngc5728",
    "ngc7172",
    "ngc7582",
]

SPIRIT_REGION_IDS = [
    "ngc4051_0",
    "ngc4051_1",
    "ngc4051_2",
    "ngc4051_3",
    "ngc4051_6",
    "ngc4051_5",
    "ngc4051_4",
    "ngc3227_0_large",
    "ngc3227_1_large",
    "ngc3227_2_large",
    "ngc3227_3_large",
    "ngc3227_4_large",
    "ngc3227_5_large",
    "ngc3227_6_large",
    "ngc5506_2_large",
    "ngc5506_0_large",
    "ngc5506_1_large",
    "ngc5506_3_large",
    "ngc5728_0_large",
    "ngc5728_1_large",
    "ngc5728_2_large",
    "ngc5728_3_large",
    "ngc5728_4_large",
    "ngc5728_1",
    "ngc5728_2",
    "ngc5728_3",
    "ngc5728_4",
    "ngc5728_5",
    "ngc5728_6",
    "ngc5728_7",
    "ngc7172_2_large",
    "ngc7172_0_large",
    "ngc7172_1_large",
    "ngc7172_3_large",
    "ngc7582_0_large",
    "ngc7582_1_large",
    "ngc7582_4_large",
    "ngc7582_0_small",
    "ngc7582_1_small",
    "ngc7582_2_large",
    "ngc7582_3_large",
    "ic5063_0_large",
    "ic5063_1_large",
    "Eso420_3",
    "Eso420_4",
    "Eso420_1",
    "Eso420_2",
    "Eso420_5",
]

TEST_OBJECTS = [
    Path("/Users/frdonnan/Downloads/Seyf_Nuclei/ngc4051"),
    Path("/Users/frdonnan/Downloads/Seyf_Nuclei/ngc5728"),
    Path("/Users/frdonnan/Documents/PAHFIT_NoGP/SPIRIT/Results/NGC 3256_Nuc1_SF1"),
]

OUTDIR = Path(__file__).resolve().parent / "PAH_FluxError_Tests"
DEFAULT_SUMMARY_CSV = OUTDIR / "Seyf_SPIRIT_PAH_flux_errors_2pct.csv"


def _obj_id_from_root(root: Path) -> str:
    return root.name


def _quick_dir(root: Path) -> Path:
    """Default CLI path: Results/<obj>/Differential/Quick."""
    return root / "Differential" / "Quick"


def _model_components_dir(root: Path) -> Path:
    return _quick_dir(root) / "Model Components"


def fnu_to_flambda(fnu: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Match IrModel continuum conversion → 10^-17 W/m^2/µm."""
    return np.asarray(fnu, float) * FNU_TO_FLAMBDA_FACTOR * C_UM_HZ / np.asarray(lam, float) ** 2


def load_fit_bundle_from_results_dir(results_dir: Path, obj: str) -> dict:
    """
    Load spectrum, full model, PAH components, and Output.csv from a finished
    Quick results directory (…/<ExtType>/Quick/).
    """
    results_dir = Path(results_dir)
    mdir = results_dir / "Model Components"
    csv_path = results_dir / f"{obj}Output.csv"

    lam, flux, flux_err = np.loadtxt(mdir / f"{obj}Spectrum.txt", unpack=True)
    wav, full_model = np.loadtxt(mdir / f"{obj}FullModel.txt", unpack=True)
    comps = np.loadtxt(mdir / f"{obj}PAHs_components.txt")
    if comps.ndim == 1:
        comps = comps.reshape(-1, 1)

    pah_rows = []
    complex_rows = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = row["Name"]
            if name == "PAH":
                pah_rows.append(
                    {
                        "name": name,
                        "rest_wav": float(row["Rest Wavelength (micron)"]),
                        "strength": float(row["Strength (10^-17 W/m^2)"]),
                        "eqw": float(row["Eqw (micron)"]),
                        "continuum": float(row["Continuum (10^-17 W/m^2/um)"]),
                    }
                )
            elif name in PAH_COMPLEXES:
                complex_rows[name] = {
                    "name": name,
                    "rest_wav": float(row["Rest Wavelength (micron)"]),
                    "strength": float(row["Strength (10^-17 W/m^2)"]),
                    "eqw": float(row["Eqw (micron)"]),
                    "continuum": float(row["Continuum (10^-17 W/m^2/um)"]),
                }

    if comps.shape[1] != len(pah_rows):
        raise RuntimeError(
            f"{obj}: PAHs_components has {comps.shape[1]} columns but "
            f"Output.csv has {len(pah_rows)} PAH Drudes"
        )

    # Model is on a fine wav grid; residuals on the data (lam) grid
    model_on_lam = interp1d(
        wav, full_model, kind="linear", bounds_error=False, fill_value=np.nan
    )(lam)
    residual = np.asarray(flux, float) - np.asarray(model_on_lam, float)

    # Mask emission lines on the residual spectrum (SetupFit algorithm)
    residual_masked = SetupFit.apply_emission_line_mask(
        lam, residual, flux_err, z=0.0
    )

    # Object root ≈ Results/<obj>/ (parent of ExtType folder)
    root = results_dir.parent.parent if results_dir.parent.name else results_dir

    return {
        "obj": obj,
        "root": root,
        "results_dir": results_dir,
        "lam": np.asarray(lam, float),
        "flux": np.asarray(flux, float),
        "flux_err": np.asarray(flux_err, float),
        "wav": np.asarray(wav, float),
        "full_model": np.asarray(full_model, float),
        "model_on_lam": np.asarray(model_on_lam, float),
        "comps_fnu": np.asarray(comps, float),
        "residual": residual,
        "residual_masked": np.asarray(residual_masked, float),
        "pah_rows": pah_rows,
        "complex_rows": complex_rows,
    }


def load_fit_bundle(root: Path) -> dict:
    """Load from Results/<obj>/Differential/Quick (CLI convenience)."""
    obj = _obj_id_from_root(root)
    return load_fit_bundle_from_results_dir(_quick_dir(root), obj)


def _sigma_lookup_from_results(results: list[dict]) -> dict[tuple, float]:
    """Map (Name, rest_wav) → σ_total for PAH Drudes and IrModel complexes."""
    lookup: dict[tuple, float] = {}
    for r in results:
        feat = r["feature"]
        sig = r.get("sigma_total")
        if sig is None or not np.isfinite(sig):
            continue
        if feat.startswith("PAH Drude"):
            wav = r.get("rest_wav_nominal")
            if wav is None:
                continue
            lookup[("PAH", float(wav))] = float(sig)
        elif feat in PAH_COMPLEXES:
            nom = PAH_COMPLEXES[feat][0]
            lookup[(feat, float(nom))] = float(sig)
    return lookup


def _match_sigma(lookup: dict[tuple, float], name: str, wav: float) -> float | None:
    key = (name, wav)
    if key in lookup:
        return lookup[key]
    for (n, w), sig in lookup.items():
        if n == name and abs(w - wav) < 1e-6:
            return sig
    return None


def apply_residual_pah_errors_to_quick_output(
    results_dir,
    obj_name: str,
    *,
    delta_c_frac: float = DELTA_C_FRAC,
    output_df=None,
):
    """
    After a Quick fit has written Output.csv + Model Components, estimate
    residual-based PAH flux uncertainties and overwrite S_err+ / S_err-
    (same value) for Name=='PAH' Drudes and PAH Complex rows only.

    Called automatically from IrModel.RunFit when binNo==0 (Quick).

    Parameters
    ----------
    results_dir : path to …/<ExtType>/Quick/
    obj_name : object ID (CSV stem)
    output_df : optional pandas DataFrame already in memory (avoids re-read)

    Returns
    -------
    pandas.DataFrame with updated PAH errors (also written to disk).
    """
    import pandas as pd

    results_dir = Path(results_dir)
    csv_path = results_dir / f"{obj_name}Output.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    bundle = load_fit_bundle_from_results_dir(results_dir, obj_name)
    lookup = _sigma_lookup_from_results(
        estimate_all_features(bundle, delta_c_frac=delta_c_frac)
    )

    if output_df is None:
        df = pd.read_csv(csv_path)
    else:
        df = output_df.copy()

    n_updated = 0
    for i, row in df.iterrows():
        name = row["Name"]
        if name != "PAH" and name not in PAH_COMPLEXES:
            continue
        try:
            wav = float(row["Rest Wavelength (micron)"])
        except (TypeError, ValueError):
            continue
        sig = _match_sigma(lookup, name, wav)
        if sig is None:
            continue
        df.at[i, "S_err+"] = float(sig)
        df.at[i, "S_err-"] = float(sig)
        n_updated += 1

    df.to_csv(csv_path, index=False)
    print(
        f"Residual PAH flux uncertainties applied to {n_updated} features "
        f"(δ_c={100 * delta_c_frac:.1f}%) → {csv_path.name}"
    )
    return df


def match_drude_indices(pah_rows: list[dict], member_cents: tuple[float, ...], atol: float = 0.05) -> list[int]:
    """Indices of Drudes whose CSV rest wavelength matches complex members."""
    idx = []
    for j, row in enumerate(pah_rows):
        if any(abs(row["rest_wav"] - c) <= atol for c in member_cents):
            idx.append(j)
    return idx


def feature_profile_fnu(bundle: dict, drude_indices: list[int]) -> np.ndarray:
    return np.sum(bundle["comps_fnu"][:, drude_indices], axis=1)


def weff_from_strength_and_peak(strength: float, profile_fnu: np.ndarray, wav: np.ndarray) -> tuple[float, float, float]:
    """
    W_eff [µm] = F / F_λ,peak with IrModel unit conversion.

    Returns (W_eff, lambda_peak, F_lambda_peak).
    """
    peak_i = int(np.nanargmax(profile_fnu))
    lam_peak = float(wav[peak_i])
    peak_fnu = float(profile_fnu[peak_i])
    if not np.isfinite(peak_fnu) or peak_fnu <= 0 or not np.isfinite(strength) or strength <= 0:
        return np.nan, lam_peak, np.nan
    peak_flambda = float(fnu_to_flambda(peak_fnu, lam_peak))
    weff = strength / peak_flambda
    return weff, lam_peak, peak_flambda


def delta_lambda(lam: np.ndarray) -> np.ndarray:
    """Centred channel widths on an irregular wavelength grid."""
    lam = np.asarray(lam, float)
    dlam = np.empty_like(lam)
    dlam[1:-1] = 0.5 * (lam[2:] - lam[:-2])
    dlam[0] = lam[1] - lam[0]
    dlam[-1] = lam[-1] - lam[-2]
    return np.abs(dlam)


def sigma_lambda_in_window(resid: np.ndarray, lam: np.ndarray, lam0: float, weff: float) -> tuple[float, np.ndarray, int]:
    """
    σ_λ = standard deviation of masked residuals in the W_eff window.

    Residuals are converted to F_λ (10^-17 W/m^2/µm) before measuring scatter
    so σ_noise shares Strength units.
    """
    half = 0.5 * weff
    win = (lam >= lam0 - half) & (lam <= lam0 + half) & np.isfinite(resid)
    n_pix = int(np.count_nonzero(win))
    if n_pix < 5:
        return np.nan, win, n_pix

    resid_fl = fnu_to_flambda(resid[win], lam[win])
    resid_fl = resid_fl[np.isfinite(resid_fl)]
    if resid_fl.size < 5:
        return np.nan, win, n_pix

    sigma = float(np.std(resid_fl, ddof=1))
    return sigma, win, n_pix


def sigma_noise_from_window(sigma_lambda: float, lam: np.ndarray, win: np.ndarray) -> float:
    """σ_noise = σ_λ * sqrt(Σ δλ_i^2) on the data grid inside the window."""
    if not np.isfinite(sigma_lambda) or sigma_lambda <= 0:
        return np.nan
    dlam = delta_lambda(lam)[win]
    dlam = dlam[np.isfinite(dlam) & (dlam > 0)]
    if dlam.size == 0:
        return np.nan
    return float(sigma_lambda * np.sqrt(np.sum(dlam**2)))


def estimate_feature_error(
    bundle: dict,
    *,
    label: str,
    strength: float,
    continuum: float,
    drude_indices: list[int],
    rest_wav_nominal: float | None = None,
    delta_c_frac: float = DELTA_C_FRAC,
) -> dict:
    empty = {
        "feature": label,
        "drude_indices": [],
        "strength": strength,
        "continuum": continuum,
        "weff_um": np.nan,
        "lam_peak": np.nan,
        "peak_flambda": np.nan,
        "sigma_lambda": np.nan,
        "sigma_noise": np.nan,
        "delta_c": np.nan,
        "sigma_cont": np.nan,
        "sigma_total": np.nan,
        "n_sigma": np.nan,
        "n_pix": 0,
        "rest_wav_nominal": rest_wav_nominal,
    }
    if not drude_indices:
        return empty

    profile = feature_profile_fnu(bundle, drude_indices)
    weff, lam_peak, peak_fl = weff_from_strength_and_peak(
        strength, profile, bundle["wav"]
    )
    sigma_lam, win, n_pix = sigma_lambda_in_window(
        bundle["residual_masked"], bundle["lam"], lam_peak, weff
    )
    sigma_noise = sigma_noise_from_window(sigma_lam, bundle["lam"], win)

    # Continuum slip: ΔC = δ_c * C, σ_cont = ΔC * W_eff
    if np.isfinite(continuum) and continuum > 0 and np.isfinite(weff) and weff > 0:
        delta_c = delta_c_frac * continuum
        sigma_cont = delta_c * weff
    else:
        delta_c = np.nan
        sigma_cont = np.nan

    terms = [t for t in (sigma_noise, sigma_cont) if np.isfinite(t) and t >= 0]
    if terms:
        sigma_total = float(np.sqrt(np.sum(np.square(terms))))
    else:
        sigma_total = np.nan

    n_sigma = (
        strength / sigma_total
        if np.isfinite(sigma_total) and sigma_total > 0 and np.isfinite(strength)
        else np.nan
    )
    return {
        "feature": label,
        "drude_indices": list(drude_indices),
        "drude_cents": [bundle["pah_rows"][i]["rest_wav"] for i in drude_indices],
        "strength": strength,
        "continuum": continuum,
        "weff_um": weff,
        "lam_peak": lam_peak,
        "peak_flambda": peak_fl,
        "sigma_lambda": sigma_lam,
        "sigma_noise": sigma_noise,
        "delta_c_frac": delta_c_frac,
        "delta_c": delta_c,
        "sigma_cont": sigma_cont,
        "sigma_total": sigma_total,
        "n_sigma": n_sigma,
        "n_pix": n_pix,
        "rest_wav_nominal": rest_wav_nominal,
        "window_mask": win,
        "profile_fnu": profile,
    }


def estimate_pah34(bundle: dict, delta_c_frac: float = DELTA_C_FRAC) -> dict:
    """
    PAH 3.4 µm aliphatic feature = sum of Drudes at 3.395 and 3.405 µm.

    Strength = sum of CSV Drude strengths.
    Continuum = mean of the two Drude continua (≈ continuum at 3.4 µm).
    Profile = summed PAHs_components columns for W_eff / peak / residual window.
    """
    idx = match_drude_indices(bundle["pah_rows"], PAH34_DRUDE_CENTS, atol=0.01)
    if len(idx) < 2:
        # Fall back to whatever matched; still attempt estimate if ≥1
        pass
    if not idx:
        return estimate_feature_error(
            bundle,
            label="PAH 3.4 (3.395+3.405)",
            strength=np.nan,
            continuum=np.nan,
            drude_indices=[],
            rest_wav_nominal=PAH34_NOMINAL_WAV,
            delta_c_frac=delta_c_frac,
        )

    rows = [bundle["pah_rows"][i] for i in idx]
    strength = float(np.nansum([r["strength"] for r in rows]))
    conts = [r["continuum"] for r in rows if np.isfinite(r["continuum"])]
    continuum = float(np.nanmean(conts)) if conts else np.nan
    return estimate_feature_error(
        bundle,
        label="PAH 3.4 (3.395+3.405)",
        strength=strength,
        continuum=continuum,
        drude_indices=idx,
        rest_wav_nominal=PAH34_NOMINAL_WAV,
        delta_c_frac=delta_c_frac,
    )


def estimate_all_features(bundle: dict, delta_c_frac: float = DELTA_C_FRAC) -> list[dict]:
    results = []
    # Individual Drudes
    for i, row in enumerate(bundle["pah_rows"]):
        results.append(
            estimate_feature_error(
                bundle,
                label=f"PAH Drude {row['rest_wav']:.3f}",
                strength=row["strength"],
                continuum=row["continuum"],
                drude_indices=[i],
                rest_wav_nominal=row["rest_wav"],
                delta_c_frac=delta_c_frac,
            )
        )
    # Complexes
    for cname, (nom, members) in PAH_COMPLEXES.items():
        crow = bundle["complex_rows"].get(cname)
        if crow is None:
            continue
        idx = match_drude_indices(bundle["pah_rows"], members)
        results.append(
            estimate_feature_error(
                bundle,
                label=cname,
                strength=crow["strength"],
                continuum=crow["continuum"],
                drude_indices=idx,
                rest_wav_nominal=nom,
                delta_c_frac=delta_c_frac,
            )
        )
    # Synthetic PAH 3.4 (not an IrModel complex row)
    results.append(estimate_pah34(bundle, delta_c_frac=delta_c_frac))
    return results


def estimate_summary_features(
    bundle: dict, delta_c_frac: float = DELTA_C_FRAC
) -> dict[str, dict]:
    """Return {short_key: estimate_feature_error result} for SUMMARY_FEATURES."""
    out: dict[str, dict] = {}
    for key, kind, spec in SUMMARY_FEATURES:
        if kind == "pah34":
            out[key] = estimate_pah34(bundle, delta_c_frac=delta_c_frac)
            continue
        crow = bundle["complex_rows"].get(spec)
        if crow is None:
            out[key] = estimate_feature_error(
                bundle,
                label=spec or key,
                strength=np.nan,
                continuum=np.nan,
                drude_indices=[],
                rest_wav_nominal=None,
                delta_c_frac=delta_c_frac,
            )
            continue
        nom, members = PAH_COMPLEXES[spec]
        idx = match_drude_indices(bundle["pah_rows"], members)
        out[key] = estimate_feature_error(
            bundle,
            label=spec,
            strength=crow["strength"],
            continuum=crow["continuum"],
            drude_indices=idx,
            rest_wav_nominal=nom,
            delta_c_frac=delta_c_frac,
        )
    return out


def write_results_csv(results: list[dict], out_csv: Path) -> None:
    fields = [
        "feature",
        "rest_wav_nominal",
        "drude_cents",
        "strength_1e-17_Wm-2",
        "continuum_1e-17_Wm-2_um-1",
        "weff_um",
        "lam_peak",
        "peak_flambda_1e-17_Wm-2_um-1",
        "sigma_lambda_1e-17_Wm-2_um-1",
        "sigma_noise_1e-17_Wm-2",
        "delta_c_frac",
        "delta_c_1e-17_Wm-2_um-1",
        "sigma_cont_1e-17_Wm-2",
        "sigma_total_1e-17_Wm-2",
        "n_sigma",
        "n_pix",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "feature": r["feature"],
                    "rest_wav_nominal": r.get("rest_wav_nominal", ""),
                    "drude_cents": ";".join(f"{c:.4f}" for c in r.get("drude_cents", [])),
                    "strength_1e-17_Wm-2": r["strength"],
                    "continuum_1e-17_Wm-2_um-1": r.get("continuum", ""),
                    "weff_um": r["weff_um"],
                    "lam_peak": r["lam_peak"],
                    "peak_flambda_1e-17_Wm-2_um-1": r["peak_flambda"],
                    "sigma_lambda_1e-17_Wm-2_um-1": r["sigma_lambda"],
                    "sigma_noise_1e-17_Wm-2": r["sigma_noise"],
                    "delta_c_frac": r.get("delta_c_frac", DELTA_C_FRAC),
                    "delta_c_1e-17_Wm-2_um-1": r.get("delta_c", ""),
                    "sigma_cont_1e-17_Wm-2": r.get("sigma_cont", ""),
                    "sigma_total_1e-17_Wm-2": r.get("sigma_total", ""),
                    "n_sigma": r["n_sigma"],
                    "n_pix": r["n_pix"],
                }
            )


def plot_complex_diagnostics(bundle: dict, result: dict, out_pdf: Path) -> None:
    """Data / model / residual view for one PAH complex."""
    if not np.isfinite(result["weff_um"]):
        print(f"Skip plot for {result['feature']}: invalid W_eff")
        return

    lam = bundle["lam"]
    wav = bundle["wav"]
    half = 0.5 * result["weff_um"]
    lam0 = result["lam_peak"]
    pad = max(0.5, 1.5 * half)
    x0, x1 = lam0 - pad, lam0 + pad

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.5), sharex=True)

    # Panel 1: spectrum + model + PAH complex
    ax = axes[0]
    m = (lam >= x0) & (lam <= x1)
    ax.plot(lam[m], bundle["flux"][m], color="k", lw=0.6, label="Data")
    ax.plot(lam[m], bundle["model_on_lam"][m], color="tab:red", lw=0.8, label="Full model")
    mw = (wav >= x0) & (wav <= x1)
    pah_fnu = result["profile_fnu"]
    # Continuum under PAH ≈ model − this is approximate; show PAH on continuum zero
    ax.plot(wav[mw], pah_fnu[mw], color="tab:purple", lw=0.9, label="PAH complex")
    ax.axvspan(lam0 - half, lam0 + half, color="tab:orange", alpha=0.2, label=r"$W_{\rm eff}$ window")
    ax.axvline(lam0, color="tab:orange", ls="--", lw=0.7)
    ax.set_ylabel(r"$F_\nu$ (fit units)")
    ax.legend(fontsize=7, loc="best")
    ax.set_title(
        f"{bundle['obj']} — {result['feature']}\n"
        f"F={result['strength']:.4g}, "
        f"W_eff={result['weff_um']:.3f} µm, "
        f"σ_noise={result['sigma_noise']:.4g}, "
        f"σ_cont={result.get('sigma_cont', np.nan):.4g} "
        f"(δ_c={100 * result.get('delta_c_frac', DELTA_C_FRAC):.0f}%), "
        f"σ_tot={result.get('sigma_total', np.nan):.4g}, "
        f"n_σ={result['n_sigma']:.2f}"
    )

    # Panel 2: residuals (raw + masked)
    ax = axes[1]
    ax.axhline(0, color="0.5", lw=0.5)
    ax.plot(lam[m], bundle["residual"][m], color="0.6", lw=0.5, label="Residual")
    ax.plot(
        lam[m],
        bundle["residual_masked"][m],
        color="tab:blue",
        lw=0.7,
        label="Residual (lines masked)",
    )
    ax.axvspan(lam0 - half, lam0 + half, color="tab:orange", alpha=0.2)
    ax.set_ylabel(r"Residual $F_\nu$")
    ax.legend(fontsize=7, loc="best")

    # Panel 3: F_λ residuals in window + σ_λ
    ax = axes[2]
    win = result["window_mask"]
    if win is not None and np.any(win):
        resid_fl = fnu_to_flambda(bundle["residual_masked"][win], lam[win])
        ax.plot(lam[win], resid_fl, color="tab:blue", lw=0.7)
        ax.axhline(result["sigma_lambda"], color="tab:red", ls="--", lw=0.8, label=rf"$\sigma_\lambda$={result['sigma_lambda']:.3g}")
        ax.axhline(-result["sigma_lambda"], color="tab:red", ls="--", lw=0.8)
        ax.axhline(0, color="0.5", lw=0.5)
    ax.axvspan(lam0 - half, lam0 + half, color="tab:orange", alpha=0.2)
    ax.set_xlabel(r"Rest wavelength (µm)")
    ax.set_ylabel(r"Residual $F_\lambda$")
    ax.legend(fontsize=7, loc="best")
    ax.set_xlim(x0, x1)

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def process_object(
    root: Path,
    outdir: Path = OUTDIR,
    *,
    make_plots: bool = True,
    write_per_object_csv: bool = True,
    delta_c_frac: float = DELTA_C_FRAC,
) -> list[dict]:
    print(f"\n=== {root} ===")
    bundle = load_fit_bundle(root)
    results = estimate_all_features(bundle, delta_c_frac=delta_c_frac)

    outdir.mkdir(parents=True, exist_ok=True)
    if write_per_object_csv:
        out_csv = outdir / f"{bundle['obj']}_PAH_flux_errors.csv"
        write_results_csv(results, out_csv)
        print(f"Wrote {out_csv}")

    # Print summary for complexes (+ PAH 3.4) and optionally plot 7.7 / 11.3
    for r in results:
        is_complex = "Complex" in r["feature"] or r["feature"].startswith("PAH 3.4")
        if not is_complex:
            continue
        print(
            f"  {r['feature']:28s}  F={r['strength']:.4g}  "
            f"W_eff={r['weff_um']:.4f} µm  "
            f"σ_noise={r['sigma_noise']:.4g}  "
            f"σ_cont={r.get('sigma_cont', np.nan):.4g}  "
            f"σ_tot={r.get('sigma_total', np.nan):.4g}  "
            f"n_σ={r['n_sigma']:.2f}  "
            f"drudes={r.get('drude_cents', [])}"
        )
        if make_plots and r["feature"] in PLOT_COMPLEXES:
            safe = r["feature"].replace(" ", "_").replace(".", "p")
            plot_complex_diagnostics(
                bundle,
                r,
                outdir / f"{bundle['obj']}_{safe}_error_diag.pdf",
            )
    return results


def resolve_object_dir(requested: str, base: Path) -> tuple[Path | None, str]:
    """
    Resolve a requested object folder under base.

    Tries exact, then case-insensitive exact, then case-insensitive prefix
    match (prefer shortest directory name among prefix hits).

    Returns (path_or_None, note).
    """
    if not base.is_dir():
        return None, f"base missing: {base}"

    exact = base / requested
    if exact.is_dir():
        return exact, "exact"

    dirs = [p for p in base.iterdir() if p.is_dir()]
    lower = requested.lower()

    ci = [p for p in dirs if p.name.lower() == lower]
    if len(ci) == 1:
        return ci[0], f"case-insensitive exact → {ci[0].name}"
    if len(ci) > 1:
        # Prefer exact-length / no suffix if ambiguous
        ci_sorted = sorted(ci, key=lambda p: (len(p.name), p.name))
        return ci_sorted[0], f"case-insensitive exact (ambiguous, chose {ci_sorted[0].name})"

    # Prefix match: requested is a prefix of dir name, or vice versa
    prefix_hits = [
        p
        for p in dirs
        if p.name.lower().startswith(lower) or lower.startswith(p.name.lower())
    ]
    # Prefer dirs that start with the request (e.g. Eso420_3 → eso420_3 not eso420_3_clustering)
    starts = [p for p in prefix_hits if p.name.lower().startswith(lower)]
    pool = starts if starts else prefix_hits
    if not pool:
        return None, "unresolved"
    pool_sorted = sorted(pool, key=lambda p: (len(p.name), p.name.lower()))
    chosen = pool_sorted[0]
    note = f"prefix match → {chosen.name}"
    if len(pool) > 1:
        note += f" (from {[p.name for p in pool_sorted[:5]]}{'...' if len(pool)>5 else ''})"
    return chosen, note


def summary_row_from_features(
    *,
    object_id: str,
    sample_group: str,
    resolved_path: Path,
    resolve_note: str,
    features: dict[str, dict],
    status: str = "ok",
    error: str = "",
) -> dict:
    row = {
        "object_id": object_id,
        "sample_group": sample_group,
        "resolved_path": str(resolved_path) if resolved_path else "",
        "resolve_note": resolve_note,
        "status": status,
        "error": error,
        "delta_c_frac": DELTA_C_FRAC,
    }
    for key, _, _ in SUMMARY_FEATURES:
        r = features.get(key, {})
        row[f"{key}_flux"] = r.get("strength", np.nan)
        row[f"{key}_err"] = r.get("sigma_total", np.nan)
        row[f"{key}_nsigma"] = r.get("n_sigma", np.nan)
        row[f"{key}_weff"] = r.get("weff_um", np.nan)
        row[f"{key}_sigma_noise"] = r.get("sigma_noise", np.nan)
        row[f"{key}_sigma_cont"] = r.get("sigma_cont", np.nan)
    return row


def summary_fieldnames() -> list[str]:
    fields = [
        "object_id",
        "sample_group",
        "resolved_path",
        "resolve_note",
        "status",
        "error",
        "delta_c_frac",
    ]
    for key, _, _ in SUMMARY_FEATURES:
        fields.extend(
            [
                f"{key}_flux",
                f"{key}_err",
                f"{key}_nsigma",
                f"{key}_weff",
                f"{key}_sigma_noise",
                f"{key}_sigma_cont",
            ]
        )
    return fields


def write_summary_csv(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = summary_fieldnames()
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for k in fields:
                v = row.get(k, "")
                if isinstance(v, float) and not np.isfinite(v):
                    out[k] = ""
                else:
                    out[k] = v
            w.writerow(out)


def process_summary_object(
    requested_id: str,
    sample_group: str,
    base: Path,
    *,
    delta_c_frac: float = DELTA_C_FRAC,
) -> dict:
    root, note = resolve_object_dir(requested_id, base)
    if root is None:
        print(f"SKIP unresolved {sample_group}/{requested_id}: {note}")
        return summary_row_from_features(
            object_id=requested_id,
            sample_group=sample_group,
            resolved_path=Path(""),
            resolve_note=note,
            features={},
            status="unresolved",
            error=note,
        )

    qdir = _quick_dir(root)
    if not qdir.is_dir():
        msg = f"missing Differential/Quick under {root}"
        print(f"SKIP {requested_id}: {msg}")
        return summary_row_from_features(
            object_id=requested_id,
            sample_group=sample_group,
            resolved_path=root,
            resolve_note=note,
            features={},
            status="missing_quick",
            error=msg,
        )

    try:
        print(f"\n=== [{sample_group}] {requested_id} → {root.name} ({note}) ===")
        bundle = load_fit_bundle(root)
        features = estimate_summary_features(bundle, delta_c_frac=delta_c_frac)
        for key, r in features.items():
            print(
                f"  {key:8s}  F={r['strength']:.4g}  "
                f"σ={r.get('sigma_total', np.nan):.4g}  "
                f"nσ={r.get('n_sigma', np.nan):.2f}"
            )
        return summary_row_from_features(
            object_id=requested_id,
            sample_group=sample_group,
            resolved_path=root,
            resolve_note=note if note != "exact" else "exact",
            features=features,
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001 — batch must not crash
        print(f"FAIL {requested_id}: {exc}")
        return summary_row_from_features(
            object_id=requested_id,
            sample_group=sample_group,
            resolved_path=root,
            resolve_note=note,
            features={},
            status="error",
            error=str(exc),
        )


def run_seyf_spirit_summary(
    out_csv: Path = DEFAULT_SUMMARY_CSV,
    *,
    delta_c_frac: float = DELTA_C_FRAC,
) -> list[dict]:
    print(f"DELTA_C_FRAC = {delta_c_frac} ({100 * delta_c_frac:.1f}%)")
    assert abs(delta_c_frac - 0.02) < 1e-12, "Expected δ_c = 0.02 for this run"
    rows: list[dict] = []

    for oid in SEYF_NUCLEI_IDS:
        rows.append(
            process_summary_object(
                oid, "Seyf_Nuclei", SEYF_NUCLEI_BASE, delta_c_frac=delta_c_frac
            )
        )
    for oid in SPIRIT_REGION_IDS:
        rows.append(
            process_summary_object(
                oid, "SPIRIT_region", SPIRIT_RESULTS_BASE, delta_c_frac=delta_c_frac
            )
        )

    write_summary_csv(rows, out_csv)
    print(f"\nWrote summary CSV: {out_csv}")

    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] != "ok"]
    print(f"Succeeded: {len(ok)} / {len(rows)}")
    print(f"Failed/skipped: {len(failed)} / {len(rows)}")
    if failed:
        for r in failed:
            print(f"  - {r['object_id']} [{r['sample_group']}] {r['status']}: {r['error']}")

    # Example rows
    examples = {"ngc4051", "ngc4051_3"}
    print("\nExample rows:")
    for r in rows:
        if r["object_id"] not in examples:
            continue
        print(
            f"  {r['object_id']}: status={r['status']} "
            f"PAH3.3={r.get('PAH3.3_flux')}±{r.get('PAH3.3_err')} "
            f"PAH3.4={r.get('PAH3.4_flux')}±{r.get('PAH3.4_err')} "
            f"PAH7.7={r.get('PAH7.7_flux')}±{r.get('PAH7.7_err')}"
        )
    return rows


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Residual-based PAH flux uncertainties (δ_c continuum term)."
    )
    p.add_argument(
        "--summary-batch",
        action="store_true",
        help="Run Seyf nuclei + SPIRIT regions and write one wide summary CSV.",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"Summary CSV path (default: {DEFAULT_SUMMARY_CSV})",
    )
    p.add_argument(
        "--delta-c-frac",
        type=float,
        default=DELTA_C_FRAC,
        help=f"Fractional continuum uncertainty δ_c (default: {DELTA_C_FRAC})",
    )
    p.add_argument(
        "--object",
        type=Path,
        action="append",
        default=None,
        help="Process a specific object root (Differential/Quick parent). Repeatable.",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip diagnostic PDF plots.",
    )
    p.add_argument(
        "--no-per-object-csv",
        action="store_true",
        help="Skip writing per-object long-format CSVs.",
    )
    return p


def main(argv: list[str] | None = None):
    args = build_argparser().parse_args(argv)

    if args.summary_batch:
        run_seyf_spirit_summary(args.out_csv, delta_c_frac=args.delta_c_frac)
        return

    roots = args.object if args.object else TEST_OBJECTS
    for root in roots:
        if not root.is_dir():
            print(f"Skip missing path: {root}")
            continue
        process_object(
            root,
            make_plots=not args.no_plots,
            write_per_object_csv=not args.no_per_object_csv,
            delta_c_frac=args.delta_c_frac,
        )


if __name__ == "__main__":
    main()
