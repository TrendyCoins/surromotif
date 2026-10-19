"""Pieces shared by the export modules (mri/export_pb.py and the legacy ones in mri/legacy/): exact decimals of float32
values, the code version of the export and the text of the 17 side inputs 'nt_PRV' (NT3_PRV_ch48 and PB_s64h24_d1)."""
import os
import subprocess

import numpy as np

from .nt3 import NT_CLIP_LOG, NT_CLIP_Z, NT_GEO
from .pine_common import fmt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _num(x):
    return float(fmt(x))


def _nums(a):
    a = np.asarray(a)
    return [_nums(v) for v in a] if a.ndim > 1 else [_num(v) for v in a]


def git_version():
    try:
        h = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', 'mri'], capture_output=True, text=True).stdout
        return h + ('+dirty' if dirty.strip() else '')
    except OSError:
        return 'unknown'


SIDE_SPEC = [
    'log_D = log(D), D = max(median |r| over the 64 bars of the window, floor of the symbol)',
    'log_mean_abs_r = log(max(mean |r| over the 64 bars of the window, floor of the symbol))',
] + [f'path_{n}{"ab"[k - 6] if n == 2 else ""}: chunk {k + 1}/8 of the path before the window ({n} bars; chunks of '
     f'{", ".join(map(str, NT_GEO))} bars over bars s-200 .. s-2, oldest first): (close of its last bar - open of its first '
     f'bar) / sqrt({n}) / rmed, clipped to +-{NT_CLIP_Z:g}, times the orientation sign' for k, n in enumerate(NT_GEO)] + [
    'range_pos = 2 * (close[s-2] - lmin) / (hmax - lmin) - 1 (0 if hmax = lmin), hmax / lmin = max high / min low over '
    'bars s-200 .. s-2 (199 bars), times the orientation sign',
    f'range_width = log(max(hmax - lmin, 1e-12) / (sqrt(199) * rmed)), clipped to +-{NT_CLIP_LOG:g}, no sign',
    f'close_dev = (close[s-2] - mean close over bars s-200 .. s-2) / (sqrt(199) * rmed), clipped to +-{NT_CLIP_Z:g}, times '
    'the orientation sign',
    f'vol_16 = log(max(mean(high - low) over bars s-16 .. s-1 / rmean, 0.001)), clipped to +-{NT_CLIP_LOG:g}, no sign',
    f'vol_64 = log(max(mean(high - low) over bars s-64 .. s-1 / rmean, 0.001)), clipped to +-{NT_CLIP_LOG:g}, no sign',
    f'range_mean_med = log(max(rmean / rmed, 0.001)), clipped to +-{NT_CLIP_LOG:g}, no sign',
    f'body_med = log(max(bmed / rmed, 0.001)), clipped to +-{NT_CLIP_LOG:g}, no sign']
