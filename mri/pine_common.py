"""Pieces shared by the Pine generators (mri/pine_pb.py and the legacy ones in mri/legacy/): the dense float32 strings
(3 CJK characters per float32, bit for bit), number literals and the offsets of the path chunks before a window.

Dense strings: each float32 — all 32 bits, exact — as 3 characters of a 2048-character alphabet (CJK ideographs
U+4E00..U+55FF, 11 bits per character); the Pine decoder is arithmetic on powers of two, so Pine holds the float32 weights
bit for bit (decode_dense is its Python mirror).
"""
import numpy as np

from .nt3 import NT_GEO

L = 64
ALPHA0, ALPHA_N = 0x4E00, 2048               # alphabet of the dense weight strings: 2048 CJK ideographs
STR_MAX = 3999                               # characters per weight string


def fmt(x):
    """Exact positional decimal of the float32 value (as float64): no exponent, safe as a Pine literal, parses back to
       exactly the float32 value."""
    s = np.format_float_positional(np.float64(np.float32(x)), unique=True, trim='-')
    return s if '.' in s else s + '.0'          # always a float literal


def fmt_d(x):
    """Positional decimal of a float64 (no exponent)."""
    s = np.format_float_positional(np.float64(x), unique=True, trim='-')
    return s if '.' in s else s + '.0'


def float_list(xs):
    return ', '.join(fmt(x) for x in xs)


def encode_dense(vals):
    """float32 -> 3 characters each: bits = d0 * 2^22 + d1 * 2^11 + d2, character = chr(ALPHA0 + d)."""
    bits = np.asarray(vals, dtype=np.float32).view(np.uint32).astype(np.int64)
    d = np.stack([bits >> 22, (bits >> 11) & 2047, bits & 2047], axis=1)
    return ''.join(chr(ALPHA0 + int(x)) for x in d.ravel())


def decode_dense(s):
    """Python mirror of the Pine decoder (arithmetic on doubles, powers of two): exact float32 values."""
    out = []
    for k in range(len(s) // 3):
        b = (ord(s[3 * k]) - ALPHA0) * 4194304.0 + (ord(s[3 * k + 1]) - ALPHA0) * 2048.0 + (ord(s[3 * k + 2]) - ALPHA0)
        sg = -1.0 if b >= 2147483648.0 else 1.0
        u = b - 2147483648.0 if b >= 2147483648.0 else b
        e = float(np.floor(u / 8388608.0))
        m = u - e * 8388608.0
        v = m * 2.0 ** -149 if e == 0 else (1.0 + m / 8388608.0) * 2.0 ** (e - 127)
        out.append(sg * v)
    return out


ALPHA_BLOCK = 'buildAlpha() =>\n    map<string, int> m = map.new<string, int>()\n' \
              '    array<string> cs = str.split("{alpha}", "")\n    for [i, c] in cs\n        map.put(m, c, i)\n    m\n'

DECODE_FLOAT = r"""        float sg = b >= 2147483648.0 ? -1.0 : 1.0
        float u = b >= 2147483648.0 ? b - 2147483648.0 : b
        float e = math.floor(u / 8388608.0)
        float m = u - e * 8388608.0
        float v = e == 0 ? m * math.pow(2.0, -149) : (1.0 + m / 8388608.0) * math.pow(2.0, e - 127)
"""


def chunk_offsets():
    """(n, offset of the chunk's last bar, offset of its first bar) from the window's last bar, oldest chunk first.
       Bar s-1 is at offset L, s-2 at L + 1; chunks run over bars s-200 .. s-2."""
    res, end = [], L + 1
    for n in NT_GEO[::-1]:
        start = end + n - 1
        res.append((n, end, start))
        end = start + 1
    assert end == L + 200
    return res[::-1]
