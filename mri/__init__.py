"""Market Randomness Index (MRI): Python reference of the SuMo classifier indicator.

Modules:
    params   - model weights, buffers, calibration (load from research npz or export JSON)
    features - per-bar inputs from OHLC (vectorized, pandas semantics of research)
    batch    - vectorized inference over a whole bar series
    stream   - bar-by-bar inference with ring buffers, exactly as the Pine indicator computes
    nt3      - NT3_PRV_ch48 (the indicator model since session 3): history inputs from OHLC, vectorized and bar-by-bar
    export_nt3, pine_nt3 - NT3 export JSON / golden file and the Pine generator (export, pine_gen: legacy F3_g)
"""

L = 64                  # window length, bars
N_NORM = 200            # bars in the rolling median of (high - low), closed='left'
BAR_MS = 5 * 60 * 1000  # 5M bars
GAP_MS = 1.5 * BAR_MS   # interval above this -> grid gap, bar invalid
BARS_DAY = 288
SPAN_SIDE = 7 * BARS_DAY    # EMA |r| for side input 1 (vol to 7-day norm)
SPAN_CTX = BARS_DAY         # EMA |r| for context normalization (1 day)
CTX_BLOCK, CTX_NBLK = 16, 8
TIE_EPS = 1e-4          # rule R1 (docs/rules.md): flip only if X_end < -TIE_EPS * D
