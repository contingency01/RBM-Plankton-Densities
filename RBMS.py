import numpy as np
from scipy.special import expit as sigmoid

class MixedRBM:
    """
    Mixed visible-unit RBM: Bernoulli presence + Gaussian log-abundance.
    Each taxon i has a binary v_b[i] (presence) and real v_g[i] (log-count).
    Hidden units are Bernoulli. Trained with CD-1.
    """

    def __init__(self, n_vis, n_hidden, sigma=1.0, lr=1e-3, momentum=0.9, l2=1e-4):
        self.nv, self.nh      = n_vis, n_hidden
        self.sigma            = sigma
        self.lr, self.mom, self.l2 = lr, momentum, l2
        rng = np.random.default_rng(42)

        self.Wb  = rng.normal(0, 0.01, (n_hidden, n_vis)).astype(np.float32)  # → binary visible
        self.Wg  = rng.normal(0, 0.01, (n_hidden, n_vis)).astype(np.float32)  # → Gaussian visible
        self.bb  = np.zeros(n_vis,    dtype=np.float32)   # bias for binary visible
        self.bg  = np.zeros(n_vis,    dtype=np.float32)   # bias for Gaussian visible
        self.bh  = np.zeros(n_hidden, dtype=np.float32)   # hidden bias

        self._dWb = np.zeros_like(self.Wb); self._dWg = np.zeros_like(self.Wg)
        self._dbb = np.zeros_like(self.bb); self._dbg = np.zeros_like(self.bg)
        self._dbh = np.zeros_like(self.bh)

    # ── Conditionals ──────────────────────────────────────────────────────────
    def h_given_v(self, vb, vg):
        return sigmoid(self.Wb @ vb + self.Wg @ vg + self.bh)

    def vb_given_h(self, h):
        return sigmoid(self.Wb.T @ h + self.bb)

    def vg_mean_given_h(self, h):
        return self.sigma**2 * (self.Wg.T @ h + self.bg)

    # ── CD-1 step ─────────────────────────────────────────────────────────────
    def cd1_step(self, vb, vg):
        s2 = self.sigma**2

        # Positive phase
        hp   = self.h_given_v(vb, vg)
        hs   = (np.random.random(self.nh) < hp).astype(np.float32)

        # Negative phase
        vb_p = self.vb_given_h(hs)
        vb_n = (np.random.random(self.nv) < vb_p).astype(np.float32)
        vg_n = self.vg_mean_given_h(hs)                   # mean-field for stability
        hn   = self.h_given_v(vb_n, vg_n)

        # Gradients
        dWb = np.outer(hp, vb)  - np.outer(hn, vb_n)
        dWg = np.outer(hp, vg)  - np.outer(hn, vg_n)
        dbb = vb  - vb_n
        dbg = vg  - vg_n
        dbh = hp  - hn

        lr = self.lr
        self._dWb = self.mom * self._dWb + lr * (dWb - self.l2 * self.Wb)
        self._dWg = self.mom * self._dWg + lr * (dWg - self.l2 * self.Wg)
        self._dbb = self.mom * self._dbb + lr * dbb
        self._dbg = self.mom * self._dbg + lr * dbg
        self._dbh = self.mom * self._dbh + lr * dbh

        self.Wb += self._dWb; self.Wg += self._dWg
        self.bb += self._dbb; self.bg += self._dbg; self.bh += self._dbh

        vg_rec = self.vg_mean_given_h(hp)
        return float(np.mean((vg - vg_rec)**2))

    def encode(self, Vb, Vg):
        return np.stack([self.h_given_v(Vb[t], Vg[t]) for t in range(len(Vg))])
