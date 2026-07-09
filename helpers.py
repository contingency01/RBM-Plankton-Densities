from pathlib import Path
import pandas as pd
import numpy as np

def find_project_dir(marker='TimeSeries_countsuL_clean.csv', folder='RBM-Plankton-Densities'):
    """Find the notebook/data directory even if the kernel starts elsewhere."""
    for base in [Path.cwd(), *Path.cwd().parents]:
        if (base / marker).exists():
            return base
        if (base / folder / marker).exists():
            return base / folder
    raise FileNotFoundError(
        f'Could not find {marker!r} from {Path.cwd()} or its parent directories.'
    )

# ── Validation MSE helper ─────────────────────────────────────────────────────
def compute_val_mse(model, X_v):
    """One-step-ahead reconstruction MSE on a held-out sequence."""
    errs = []
    for t in range(1, len(X_v)):
        hp    = model.h_given_v(X_v[t], X_v[t-1])
        v_rec = model.v_given_h(hp, X_v[t-1])
        errs.append(float(np.mean((X_v[t] - v_rec) ** 2)))
    return float(np.mean(errs))

# ── Generate synthetic communities and compare RBM vs real data ─────────────
def sample_gaussian_rbm(model, n_samples, burn_in=1000, thin=20, seed=42):
    rng_local = np.random.default_rng(seed)
    v = rng_local.normal(0, 1, model.nv).astype(np.float32)
    samples = []
    total_steps = burn_in + n_samples * thin

    for step in range(total_steps):
        hp = model.h_given_v(v)
        hs = (rng_local.random(model.nh) < hp).astype(np.float32)
        v_mean = model.v_given_h(hs).astype(np.float32)
        v = v_mean 

        if step >= burn_in and (step - burn_in) % thin == 0:
            samples.append(v.copy())

    return np.stack(samples)


def read_data():
    PROJECT_DIR    = find_project_dir()
    CSV_PATH       = PROJECT_DIR / 'TimeSeries_countsuL_clean.csv'
    df = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    df = df.dropna(how='all').fillna(0)
    
    MIN_PREVALENCE = 0.05
    presence  = (df > 0).mean(axis=0)
    keep_prev = presence >= MIN_PREVALENCE
    
    eps_all   = df[df > 0].min().min() * 0.1
    X_all_log = np.log10(df.values + eps_all)
    log_std   = X_all_log.std(axis=0)
    MIN_LOG_STD = 0.5
    keep_var  = log_std >= MIN_LOG_STD
    
    keep    = keep_prev & keep_var
    df      = df.loc[:, keep]
    species = df.columns.tolist()
    
    real_den = df.values.astype(np.float32)               # raw counts µL⁻¹
 
    eps   = df[df > 0].min().min() * 0.1
    X_raw = np.log10(df.values + eps).astype(np.float32)
    
    mu  = X_raw.mean(0, keepdims=True)
    sig = X_raw.std(0, keepdims=True) + 1e-8
    X   = (X_raw - mu) / sig
    
    dates    = df.index
    return X, X_raw, real_den, dates, species