import numpy as np
import numpy.random as rng
from scipy.special import expit as sigmoid

class ConditionalRBM:
    """
    Gaussian-visible / Bernoulli-hidden Conditional RBM.
    Autoregressive order = 1 (conditions on previous time step only).
    Trained with Contrastive Divergence k=1 (CD-1).
    """

    def __init__(self, n_visible, n_hidden, lr=1e-3, momentum=0.9, l2=1e-4):
        self.nv, self.nh = n_visible, n_hidden
        self.lr, self.mom, self.l2 = lr, momentum, l2

        # RBM weights + biases
        self.W  = rng.normal(0, 0.01, (n_hidden, n_visible)).astype(np.float32)
        self.bh = np.zeros(n_hidden,  dtype=np.float32)
        self.bv = np.zeros(n_visible, dtype=np.float32)

        # Autoregressive weights  (past visible → dynamic biases)
        self.A  = rng.normal(0, 0.01, (n_hidden,  n_visible)).astype(np.float32)  # → hidden bias
        self.B  = rng.normal(0, 0.01, (n_visible, n_visible)).astype(np.float32)  # → visible bias

        # Momentum buffers
        self._dW = np.zeros_like(self.W)
        self._dbh = np.zeros_like(self.bh)
        self._dbv = np.zeros_like(self.bv)
        self._dA  = np.zeros_like(self.A)
        self._dB  = np.zeros_like(self.B)

    # ── Conditional distributions ─────────────────────────────────────────────
    def _bh_dyn(self, v_prev):  return self.bh + self.A @ v_prev
    def _bv_dyn(self, v_prev):  return self.bv + self.B @ v_prev

    def h_given_v(self, v, v_prev):
        """P(h=1 | v_t, v_{t-1})  — shape (n_hidden,)"""
        return sigmoid(self.W @ v + self._bh_dyn(v_prev))

    def v_given_h(self, h, v_prev):
        """Mean of P(v_t | h_t, v_{t-1}) — Gaussian mean"""
        return self.W.T @ h + self._bv_dyn(v_prev)

    # ── CD-1 step ─────────────────────────────────────────────────────────────
    def cd1_step(self, v, v_prev):
        # Positive phase
        hp = self.h_given_v(v, v_prev)
        hs = (rng.random(self.nh) < hp).astype(np.float32)   # sample

        # Negative phase (Gibbs step)
        v_neg = self.v_given_h(hs, v_prev)                    # Gaussian mean
        hn    = self.h_given_v(v_neg, v_prev)

        # Gradients
        dW  = np.outer(hp, v) - np.outer(hn, v_neg)
        dbh = hp - hn
        dbv = v  - v_neg
        dA  = np.outer(hp - hn, v_prev)
        dB  = np.outer(v  - v_neg, v_prev)

        # Momentum + L2
        lr = self.lr
        self._dW  = self.mom * self._dW  + lr * (dW  - self.l2 * self.W)
        self._dbh = self.mom * self._dbh + lr * dbh
        self._dbv = self.mom * self._dbv + lr * dbv
        self._dA  = self.mom * self._dA  + lr * (dA  - self.l2 * self.A)
        self._dB  = self.mom * self._dB  + lr * (dB  - self.l2 * self.B)

        self.W  += self._dW;  self.bh += self._dbh
        self.bv += self._dbv; self.A  += self._dA;  self.B += self._dB

        return float(np.mean((v - v_neg) ** 2))   # reconstruction MSE

    # ── Encode full sequence → hidden activations ─────────────────────────────
    def encode(self, X):
        H = np.zeros((len(X) - 1, self.nh), dtype=np.float32)
        for t in range(1, len(X)):
            H[t-1] = self.h_given_v(X[t], X[t-1])
        return H

    # ── Reconstruct sequence ──────────────────────────────────────────────────
    def reconstruct(self, X):
        V_rec = np.zeros_like(X[1:])
        for t in range(1, len(X)):
            hp     = self.h_given_v(X[t], X[t-1])
            V_rec[t-1] = self.v_given_h(hp, X[t-1])
        return V_rec

class GaussianRBM:
    """Gaussian-visible / Bernoulli-hidden RBM.  No temporal connections."""

    def __init__(self, n_visible, n_hidden, lr=1e-3, momentum=0.9, l2=1e-4):
        self.nv, self.nh     = n_visible, n_hidden
        self.lr, self.mom, self.l2 = lr, momentum, l2
        rng = np.random.default_rng(42)
        self.W  = rng.normal(0, 0.01, (n_hidden, n_visible)).astype(np.float32)
        self.bh = np.zeros(n_hidden,  dtype=np.float32)
        self.bv = np.zeros(n_visible, dtype=np.float32)
        self._dW  = np.zeros_like(self.W)
        self._dbh = np.zeros_like(self.bh)
        self._dbv = np.zeros_like(self.bv)

    def h_given_v(self, v):
        return sigmoid(self.W @ v + self.bh)

    def v_given_h(self, h):
        return self.W.T @ h + self.bv

    def encode(self, X):
        return np.stack([self.h_given_v(X[t]) for t in range(len(X))])


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
