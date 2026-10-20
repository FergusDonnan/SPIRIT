"""Fit every spaxel of a cube with SPIRIT.

Set CUBE_PATH and REDSHIFT below. The cube is extension 1. The wavelength
array is extension 2, in the observed frame, and is divided by (1 + redshift)
here. The fitter is then called with z = 0, because SetupFit would divide
by (1 + z) again.

Four chains run at once, one per CPU core. Inside a chain the spaxels are
neighbours, and each fit starts from the parameter vector of the previous
spaxel. Results go in Results/<cube>/<spaxel>/. The per-component PAH file
is removed after the fit; the total PAH spectrum is kept.

No error cube is read. Each spaxel gets an error spectrum from the same
estimate SetupFit uses for the fit: the smoothed standard deviation of the
spectrum about a median filter.

One progress bar in the parent counts every spaxel across the four chains.
Each chain's own printout goes to Results/<cube>/logs/chain_XX.log.

A spaxel is skipped when its fit outputs are already on disk, so a stopped
run continues from the first missing spaxel in each chain. Python memory is
released after every spaxel, and the JAX cache is cleared every CLEAR_EVERY
fits.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys

import numpy as np
from astropy.io import fits

SPIRIT_DIR = os.path.dirname(os.path.abspath(__file__))
if SPIRIT_DIR not in sys.path:
    sys.path.insert(0, SPIRIT_DIR)

# Observed-frame cube. Extension 1 is the data cube, extension 2 is wavelength.
CUBE_PATH = ""
REDSHIFT = 0.009316

N_WORKERS = 4
MIN_FINITE_FRAC = 0.5
SKIP_EXISTING = True
# JAX keeps compiled copies. Clearing every fit is safest and slowest.
CLEAR_EVERY = 5

FIT = dict(
    z=0.0,
    lam_range=[1.5, 28.0],
    binNo=0,
    useMCMC=False,
    ExtType_="Differential",
    Ices_6micron=False,
    InitialFit=False,
    BootStrap=False,
    N_bootstrap=50,
    HI_ratios="Case B",
    show_progress=False,
    N_MCMC=2000,
    N_BurnIn=10000,
    ExtCurve="D23ExtCurve",
    EmCurve="D24Emissivity",
    MIR_CH="CHExt_v3",
    NIR_CH="CH_NIR",
    Fit_NIR_CH=False,
    NIR_Ice="NIR_Ice",
    NIR_CO2="NIR_CO2",
    RegStrength=1000,
    Cont_Only=False,
    St_Cont=True,
    Extend=False,
    Fit_CO=False,
    Disable_Ices=False,
    spec_res="h",
    save_pah_components=False,
)


def _spaxel_name(y, x):
    return f"y{y:03d}_x{x:03d}"


def _params_path(cube_label, name):
    return os.path.join(
        SPIRIT_DIR,
        "Results",
        cube_label,
        name,
        FIT["ExtType_"],
        "Quick" if FIT["binNo"] == 0 else ("MCMC" if FIT["binNo"] == 1 else "Bootstrap"),
        "params.txt",
    )


def _output_path(cube_label, name):
    return os.path.join(os.path.dirname(_params_path(cube_label, name)), f"{name}Output.csv")


def _fit_is_done(cube_label, name):
    """params.txt is written mid-fit. The output table is written at the end."""
    params_file = _params_path(cube_label, name)
    return (
        os.path.isfile(params_file)
        and os.path.getsize(params_file) > 0
        and os.path.isfile(_output_path(cube_label, name))
    )


def _load_params(path):
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        return arr
    return arr[:, 1]


def _release_memory(fitted_count):
    import gc

    import matplotlib.pyplot as plt

    plt.close("all")
    gc.collect()
    if fitted_count > 0 and fitted_count % CLEAR_EVERY == 0:
        import jax

        jax.clear_caches()
        gc.collect()


def _read_cube(path, redshift):
    """Extension 1 is the cube. Extension 2 is observed-frame wavelength."""
    with fits.open(path) as hdul:
        if len(hdul) < 3:
            raise SystemExit(
                "Need the cube in extension 1 and the wavelength array in extension 2."
            )
        sci = np.asarray(hdul[1].data, dtype=np.float64)
        wave = np.asarray(hdul[2].data)
    if getattr(wave.dtype, "names", None):
        wave = np.asarray(wave[wave.dtype.names[0]])
    wave = np.ravel(wave).astype(np.float64)
    matches = np.where(np.asarray(sci.shape) == wave.size)[0]
    if matches.size == 0:
        raise SystemExit(
            f"Wavelength array has {wave.size} elements, cube shape is {sci.shape}."
        )
    sci = np.moveaxis(sci, int(matches[0]), 0)
    wave = wave / (1.0 + redshift)
    return sci, wave


def _noise_spectrum(flux):
    """Same error estimate SetupFit uses after it has scaled the spectrum."""
    import scipy.ndimage
    import scipy.signal

    flux = np.asarray(flux, dtype=np.float64)
    n = flux.size
    y_filt = scipy.signal.medfilt(np.copy(flux), 21)
    binsize = 100
    n_bins = int(n / binsize)
    if n_bins < 1:
        sigma = float(np.std(flux - y_filt))
        return np.full(n, sigma if sigma > 0 else 1e-8)
    noise = np.empty(n)
    for i in range(n_bins):
        sl = slice(i * binsize, (i + 1) * binsize)
        noise[sl] = np.std(flux[sl] - y_filt[sl])
    noise[-binsize:] = noise[-binsize - 1]
    p_fit = scipy.signal.medfilt(noise, binsize * 10 + 1)
    p_fit = scipy.ndimage.gaussian_filter1d(p_fit, 250)
    floor = 1e-8 * np.median(np.abs(flux))
    return np.maximum(p_fit, floor if floor > 0 else 1e-8)


def _valid_spaxels(sci):
    finite = np.isfinite(sci)
    frac = finite.mean(axis=0)
    ys, xs = np.where(frac >= MIN_FINITE_FRAC)
    # Serpentine so consecutive spaxels in a chain are neighbours.
    order = np.lexsort((np.where(ys % 2 == 0, xs, -xs), ys))
    return [(int(ys[i]), int(xs[i])) for i in order]


def _log_to_file(path):
    """Send this process's stdout and stderr to a chain log."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log_fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, 1)
    os.dup2(log_fd, 2)
    os.close(log_fd)
    sys.stdout = open(1, "w", buffering=1, closefd=False)
    sys.stderr = open(2, "w", buffering=1, closefd=False)


def _fit_chain(task):
    cube_path, cube_label, coords, redshift, chain_id, progress = task
    log_path = os.path.join(
        SPIRIT_DIR, "Results", cube_label, "logs", f"chain_{chain_id:02d}.log"
    )
    _log_to_file(log_path)

    import IrModel

    sci, wave = _read_cube(cube_path, redshift)

    def _note(name, y, x, status):
        report.append((name, y, x, status))
        progress.put((name, status))

    init = None
    report = []
    fitted_count = 0
    for y, x in coords:
        name = _spaxel_name(y, x)
        params_file = _params_path(cube_label, name)
        if SKIP_EXISTING and _fit_is_done(cube_label, name):
            try:
                init = _load_params(params_file)
            except Exception as exc:
                print(f"Could not read {name} params, refitting: {exc}", flush=True)
            else:
                print(f"Skipping existing {name}", flush=True)
                _note(name, y, x, "skipped")
                continue
        elif os.path.isfile(params_file):
            try:
                init = _load_params(params_file)
                print(f"{name} has parameters but no output table, refitting", flush=True)
            except Exception as exc:
                print(f"Could not read partial {name} params: {exc}", flush=True)

        flux = sci[:, y, x]
        ok = np.isfinite(wave) & np.isfinite(flux)
        if ok.sum() < 20:
            print(f"Skipping {name}: too few valid pixels", flush=True)
            _note(name, y, x, "too few pixels")
            _release_memory(fitted_count)
            continue

        flux = flux[ok]
        specdata = [wave[ok], flux, _noise_spectrum(flux)]
        print(f"Fitting {name} (warm start: {init is not None})", flush=True)
        try:
            IrModel.RunFit(
                name,
                specdata,
                init_values=init,
                results_name=f"{cube_label}/{name}",
                **FIT,
            )
            init = _load_params(params_file)
            print(f"Finished {name}", flush=True)
            _note(name, y, x, "ok")
        except Exception as exc:
            print(f"FAILED {name}: {exc}", flush=True)
            _note(name, y, x, f"FAILED {exc}")
            init = None
        finally:
            fitted_count += 1
            del flux, specdata
            _release_memory(fitted_count)
    return report


def main():
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from queue import Empty

    from tqdm import tqdm

    cube_label = os.path.splitext(os.path.basename(CUBE_PATH))[0]
    sci, wave = _read_cube(CUBE_PATH, REDSHIFT)
    print(f"{CUBE_PATH}", flush=True)
    print(
        f"redshift {REDSHIFT}, rest wavelength {wave[0]:.4f} to {wave[-1]:.4f} um",
        flush=True,
    )

    coords = _valid_spaxels(sci)
    del sci, wave
    if not coords:
        raise SystemExit("No spaxels passed the finite-fraction cut.")
    n_done = sum(
        _fit_is_done(cube_label, _spaxel_name(y, x)) for y, x in coords
    )
    print(
        f"{n_done} spaxels already fitted, {len(coords) - n_done} remaining",
        flush=True,
    )

    n_workers = min(N_WORKERS, len(coords))
    # Contiguous blocks keep the warm start on a neighbouring spaxel.
    edges = np.linspace(0, len(coords), n_workers + 1, dtype=int)
    chunks = [coords[edges[i]:edges[i + 1]] for i in range(n_workers)]
    chunks = [c for c in chunks if c]

    log_dir = os.path.join(SPIRIT_DIR, "Results", cube_label, "logs")
    print(f"{len(coords)} spaxels in {len(chunks)} chains", flush=True)
    print(f"Per-chain logs: {log_dir}", flush=True)

    ctx = mp.get_context("spawn")
    # A manager queue can be passed into the pool. A plain process queue cannot:
    # the pool pickles task arguments, and that queue only travels by inheritance.
    manager = ctx.Manager()
    try:
        progress = manager.Queue()
        tasks = [
            (CUBE_PATH, cube_label, chunk, REDSHIFT, chain_id, progress)
            for chain_id, chunk in enumerate(chunks)
        ]
        reports = []
        n_ok = n_skip = n_short = n_fail = 0
        finished = 0
        with ProcessPoolExecutor(max_workers=len(chunks), mp_context=ctx) as pool:
            futures = [pool.submit(_fit_chain, task) for task in tasks]
            with tqdm(total=len(coords), unit="spaxel", desc="Cube") as bar:
                while finished < len(coords):
                    try:
                        name, status = progress.get(timeout=5)
                    except Empty:
                        if all(future.done() for future in futures):
                            break
                        continue
                    finished += 1
                    if status == "ok":
                        n_ok += 1
                    elif status == "skipped":
                        n_skip += 1
                    elif status == "too few pixels":
                        n_short += 1
                    elif str(status).startswith("FAILED"):
                        n_fail += 1
                        tqdm.write(f"{name}: {status}")
                    bar.set_postfix(
                        ok=n_ok, skip=n_skip, short=n_short, fail=n_fail, refresh=False
                    )
                    bar.update(1)
                if finished < len(coords):
                    tqdm.write(
                        f"Progress stopped at {finished}/{len(coords)} because a chain exited early."
                    )
            for future in futures:
                try:
                    reports.extend(future.result())
                except Exception as exc:
                    tqdm.write(f"A chain stopped: {exc}")
    finally:
        manager.shutdown()

    out_dir = os.path.join(SPIRIT_DIR, "Results", cube_label)
    os.makedirs(out_dir, exist_ok=True)
    index = os.path.join(out_dir, "spaxels.txt")
    with open(index, "w") as fh:
        fh.write("name y x status\n")
        for name, y, x, status in sorted(reports):
            fh.write(f"{name} {y} {x} {status}\n")
    n_ok = sum(status == "ok" for *_, status in reports)
    n_skip = sum(status == "skipped" for *_, status in reports)
    n_fail = sum(str(status).startswith("FAILED") for *_, status in reports)
    print(f"Done. fitted {n_ok}, skipped {n_skip}, failed {n_fail}. Index: {index}", flush=True)


if __name__ == "__main__":
    main()
