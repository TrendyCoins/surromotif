"""Market Randomness Index (MRI): Python reference of the SuMo classifier indicator (model PB_s64h24_d1, line M of
session 5). Everything the Pine indicator computes is here bar by bar and vectorized, plus its export and generator.

Modules:
    params        model weights and buffers (from research npz or the export JSON)
    features      per-bar inputs from OHLC (vectorized, pandas semantics of research)
    batch         vectorized conv streams and head (shared by the network of nt3 / pb)
    nt3           the network and its inputs: history of the 200 bars before a window, vectorized and bar by bar
                  (NT3_PRV_ch48 of session 3; PB_s64h24_d1 is the same network with separable layers 2..4)
    pb            PB_s64h24_d1: one floor FLOOR, vectorized chart and the bar-by-bar stream with the line M
    linem         line M: ruler, live part by formula A, log M, display
    export_pb     export JSON (export/mri_model_pb.json) and the golden file (tests/golden/btc5m_fresh_pb.csv.gz)
    pine_pb       Pine generator of pine/mri.pine (production) and pine/proto_pb.pine (prototype with debug mode)
    pine_common, export_common   pieces shared with the legacy generators / exports
    research_bridge, research5, ceil_export   checks only: bars -> research data files, the research chain of session 5,
                  the Colab export of the SuMo_Class_Ceil cache
    legacy/       F3_g, NT3_PRV_ch48 and the EMA line of session 4 (records with their tests in tests/legacy/)
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
