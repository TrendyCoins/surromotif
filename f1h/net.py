"""The network of SP_l025_d1 (research/sumo_class_train.py CNNBase with the prod a config) over windows, numpy float64.

Input of a window ending at the bar T (research SeriesData.gw 'multi_g' -> to_shapes -> CNNBase.conv_maps):
    D     = max(median |r| over the 64 bars of the window, floor)
    x_end = end of the centered close path: cs[63] - mean(cs), cs = cumsum of r over the window
    sgn   = -1 if x_end < -orient_eps * D else +1                   (research ORIENT_EPS, rule R1)
    the 63 places t = T-62 .. T (the first bar of the window is lost by the steps of the shape):
          [sgn r_t, sgn step_hl2_t, sgn step_hlc3_t] / D / s_sd and (g_t / D - lvl_mu) / lvl_sd
    (to_shapes: the cumsum of each path, centered, flipped by sgn, divided by D; its steps are exactly sgn * value / D)
Layers: 4 x Conv1d(kernel 5, dilation d, zero padding 2 d, bias) + ReLU, d = 1, 2, 4, 8 (64 channels, 63 places each).
Pooling over the 63 places: mean, max, last (192). Side inputs (17, nt_PRV): log D, log max(mean |r| of the window, floor),
the 15 history inputs (mri-free copy in f1h.bars), the signed ones times sgn; standardized (s - side_mu) / side_sd.
Head: relu(head0_w @ [pooled, side] + head0_b) -> head3_w @ . + head3_b.
Mirror symmetry (cnn_sym, research mirror_inputs): logit = (g(window) + g(mirror)) / 2; the mirror has the path channels and
the signed side inputs negated (the level g is kept). So the logit does not depend on the orientation: g(+) and g(-) swap.
"""
import numpy as np

from .bars import history

BATCH = 256                       # windows per block (2 passes each): ~60 MB of float64 activations


def window_inputs(model, f, ohlc, T):
    """Per-window inputs of windows ending at the rows T: dict D, x_end, sgn, path (m, 63, 3), lvl (m, 63), side_raw (m, 17)
       — side_raw with the orientation sign, as research side_feats."""
    L = model.window
    T = np.asarray(T, dtype=np.int64)
    idx = T[:, None] + np.arange(-(L - 1), 1)[None, :]                  # (m, 64) rows of the window
    rw = f['r'][idx]
    D = np.maximum(np.median(np.abs(rw), axis=1), model.floor)
    cs = np.cumsum(rw, axis=1)
    x_end = cs[:, -1] - cs.mean(axis=1)
    sgn = np.where(x_end < -model.orient_eps * D, -1.0, 1.0)
    p = idx[:, 1:]                                                       # the 63 places
    path = np.stack([f['r'][p], f['step_hl2'][p], f['step_hlc3'][p]], axis=2)
    lvl = f['g'][p]
    hist = history(*ohlc, T - (L - 1), model.n_hist)
    side_sign = np.where(model.side_signed[2:][None, :], sgn[:, None], 1.0)
    side_raw = np.concatenate([np.log(D)[:, None], np.log(np.maximum(np.abs(rw).mean(axis=1), model.floor))[:, None],
                               hist * side_sign], axis=1)
    return {'D': D, 'x_end': x_end, 'sgn': sgn, 'path': path, 'lvl': lvl, 'side_raw': side_raw}


def conv_layers(model, x):
    """x: (63, B, 4) places-first -> (63, B, 64) after the 4 layers."""
    P = x.shape[0]
    for w, b, d in zip(model.conv_w, model.conv_b, model.dilations):
        pad = 2 * d                                                      # (kernel - 1) / 2 * d
        xp = np.zeros((P + 2 * pad,) + x.shape[1:])
        xp[pad:pad + P] = x
        y = np.broadcast_to(b, (P * x.shape[1], len(b))).copy()
        for j in range(w.shape[2]):                                      # tap j sees the place t - pad + j d
            tap = np.ascontiguousarray(w[:, :, j].T)                    # contiguous: numpy < 2 needs it for BLAS
            y += xp[j * d:j * d + P].reshape(-1, x.shape[2]) @ tap
        x = np.maximum(y, 0.0).reshape(P, x.shape[1], len(b))
    return x


def head(model, feats):
    z = np.maximum(feats @ model.head0_w.T + model.head0_b, 0.0)
    return (z @ model.head3_w.T)[:, 0] + model.head3_b[0]


def g_pass(model, inp, sigma):
    """g of the windows with the path channels and the signed side inputs times sigma (per window, +-1): sigma = sgn is
       the window as research orients it, sigma = -sgn its mirror."""
    D = inp['D'][:, None, None]
    path = inp['path'] * sigma[:, None, None] / D / model.s_sd
    lvl = (inp['lvl'][:, :, None] / D - model.lvl_mu) / model.lvl_sd
    x = np.concatenate([path, lvl], axis=2).transpose(1, 0, 2)           # (63, m, 4)
    h = conv_layers(model, x)
    pooled = np.concatenate([h.mean(axis=0), h.max(axis=0), h[-1]], axis=1)
    flip = np.where(model.side_signed[None, :], (sigma * inp['sgn'])[:, None], 1.0)
    side = (inp['side_raw'] * flip - model.side_mu) / model.side_sd
    return head(model, np.concatenate([pooled, side], axis=1))


def logits(model, f, ohlc, T, batch=BATCH, orient=None):
    """Logit of the windows ending at the rows T (research CNNBase.forward with sym). orient — tests only: force the
       orientation sign of every window (+1 / -1) instead of the tie rule."""
    T = np.asarray(T, dtype=np.int64)
    out = np.empty(len(T))
    for a in range(0, len(T), batch):
        inp = window_inputs(model, f, ohlc, T[a:a + batch])
        if orient is not None:
            s = np.broadcast_to(np.asarray(orient, dtype=np.float64), inp['sgn'].shape)
            inp['side_raw'][:, 2:] *= np.where(model.side_signed[2:][None, :], (s * inp['sgn'])[:, None], 1.0)
            inp['sgn'] = s.copy()
        g_or = g_pass(model, inp, inp['sgn'])
        g_mi = g_pass(model, inp, -inp['sgn'])
        out[a:a + batch] = 0.5 * (g_or + g_mi)
    return out
