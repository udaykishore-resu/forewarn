"""Small inline-SVG chart helpers for the validation report.

No plotting dependency: the report is one self-contained HTML file that a model
validator can open offline.  Colours come from CSS custom properties defined in
the report, so the same marks work in light and dark mode.  Every chart sits
next to a table with the same numbers.
"""
from __future__ import annotations

import math
from html import escape

SERIES = ["s1", "s2", "s3"]


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10) * mag
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _legend(names: list[str], x: float, y: float) -> str:
    parts, cx = [], x
    for i, n in enumerate(names):
        parts.append(f'<rect class="{SERIES[i]} fill" x="{cx:.0f}" y="{y - 9:.0f}" width="12" height="12" rx="3"/>'
                     f'<text class="lbl" x="{cx + 17:.0f}" y="{y + 1:.0f}">{escape(n)}</text>')
        cx += 17 + 7.0 * len(n) + 18
    return "".join(parts)


def _frame(w: int, h: int, body: str, label: str) -> str:
    return (f'<svg class="chart" viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}" '
            f'preserveAspectRatio="xMinYMin meet">{body}</svg>')


def line_chart(x_labels: list[str], series: list[tuple[str, list[float]]], label: str, y_fmt=lambda v: f"{v:g}",
               w: int = 640, h: int = 250, y_range: tuple[float, float] | None = None, x_title: str = "") -> str:
    ml, mr, mt, mb = 48, 110, 34, 40
    vals = [v for _, ys in series for v in ys if v is not None and not math.isnan(v)]
    lo, hi = y_range or (min(vals), max(vals))
    pad = (hi - lo) * 0.08 or 0.01
    lo, hi = (lo - pad, hi + pad) if y_range is None else (lo, hi)
    ticks = _ticks(lo, hi)
    sx = lambda i: ml + (w - ml - mr) * (i / max(len(x_labels) - 1, 1))  # noqa: E731
    sy = lambda v: mt + (h - mt - mb) * (1 - (v - lo) / (hi - lo))  # noqa: E731
    out = [_legend([n for n, _ in series], ml, 14)] if len(series) > 1 else []
    for t in ticks:
        out.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                   f'<text class="tick" x="{ml - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{escape(y_fmt(t))}</text>')
    step = max(1, len(x_labels) // 9)
    for i, xl in enumerate(x_labels):
        if i % step == 0 or i == len(x_labels) - 1:
            out.append(f'<text class="tick" x="{sx(i):.1f}" y="{h - mb + 16}" text-anchor="middle">{escape(xl)}</text>')
    if x_title:
        out.append(f'<text class="tick" x="{(ml + w - mr) / 2:.0f}" y="{h - 6}" text-anchor="middle">{escape(x_title)}</text>')
    ends = []
    for k, (name, ys) in enumerate(series):
        pts = [(sx(i), sy(v)) for i, v in enumerate(ys) if v is not None and not math.isnan(v)]
        out.append(f'<polyline class="{SERIES[k]} line" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
        for i, v in enumerate(ys):
            if v is None or math.isnan(v):
                continue
            out.append(f'<circle class="{SERIES[k]} dot" cx="{sx(i):.1f}" cy="{sy(v):.1f}" r="4">'
                       f'<title>{escape(name)} · {escape(x_labels[i])}: {escape(y_fmt(v))}</title></circle>')
        ends.append([pts[-1][1], f"{name} {y_fmt(ys[-1])}"])
    ends.sort()
    for i in range(1, len(ends)):                      # keep end labels from overlapping
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    for y, text in ends:
        out.append(f'<text class="lbl" x="{w - mr + 10}" y="{y + 4:.1f}">{escape(text)}</text>')
    return _frame(w, h, "".join(out), label)


def _col(x: float, y: float, bw: float, bh: float, cls: str, tip: str) -> str:
    r = min(4.0, bw / 2, max(bh, 0.01))
    if bh <= 0.5:
        return ""
    d = (f"M{x:.1f},{y + bh:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + bw - r:.1f} "
         f"Q{x + bw:.1f},{y:.1f} {x + bw:.1f},{y + r:.1f} V{y + bh:.1f} Z")
    return f'<path class="{cls} fill" d="{d}"><title>{escape(tip)}</title></path>'


def column_chart(categories: list[str], series: list[tuple[str, list[float]]], label: str, y_fmt=lambda v: f"{v:g}",
                 w: int = 640, h: int = 260, y_max: float | None = None, x_title: str = "") -> str:
    ml, mr, mt, mb = 48, 16, 34, 44
    hi = y_max or max(v for _, ys in series for v in ys) * 1.12
    sy = lambda v: mt + (h - mt - mb) * (1 - v / hi)  # noqa: E731
    band = (w - ml - mr) / len(categories)
    bw = min(24.0, (band - 12) / len(series) - 2)
    out = [_legend([n for n, _ in series], ml, 14)] if len(series) > 1 else []
    for t in _ticks(0, hi):
        out.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                   f'<text class="tick" x="{ml - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{escape(y_fmt(t))}</text>')
    for i, c in enumerate(categories):
        x0 = ml + band * i + (band - (bw + 2) * len(series) + 2) / 2
        for k, (name, ys) in enumerate(series):
            x = x0 + k * (bw + 2)
            out.append(_col(x, sy(ys[i]), bw, sy(0) - sy(ys[i]), SERIES[k], f"{name} · {c}: {y_fmt(ys[i])}"))
            if k == 0:
                out.append(f'<text class="val" x="{x + bw / 2:.1f}" y="{sy(ys[i]) - 5:.1f}" text-anchor="middle">{escape(y_fmt(ys[i]))}</text>')
        out.append(f'<text class="tick" x="{ml + band * (i + .5):.1f}" y="{h - mb + 16}" text-anchor="middle">{escape(c)}</text>')
    out.append(f'<line class="axis" x1="{ml}" x2="{w - mr}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    if x_title:
        out.append(f'<text class="tick" x="{(ml + w - mr) / 2:.0f}" y="{h - 6}" text-anchor="middle">{escape(x_title)}</text>')
    return _frame(w, h, "".join(out), label)


def bar_chart(labels: list[str], values: list[float], label: str, fmt=lambda v: f"{v:g}", ci: list[tuple[float, float]] | None = None,
              refs: list[tuple[float, str]] | None = None, w: int = 640, label_w: int = 190, highlight: int | None = None) -> str:
    """Horizontal bars from a zero baseline, optional interval whiskers and reference lines."""
    row, mt, mb, mr = 30, 12, 26, 78
    h = mt + mb + row * len(labels)
    lows = [c[0] for c in ci] if ci else values
    highs = [c[1] for c in ci] if ci else values
    lo = min(0.0, min(lows), *(r[0] for r in refs or [(0, "")]))
    hi = max(0.0, max(highs), *(r[0] for r in refs or [(0, "")]))
    span = (hi - lo) or 1.0
    lo, hi = lo - 0.04 * span * (lo < 0), hi + 0.04 * span
    sx = lambda v: label_w + (w - label_w - mr) * ((v - lo) / (hi - lo))  # noqa: E731
    out = []
    for t in _ticks(lo, hi, 5):
        out.append(f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{mt}" y2="{h - mb}"/>'
                   f'<text class="tick" x="{sx(t):.1f}" y="{h - mb + 15}" text-anchor="middle">{escape(fmt(t))}</text>')
    for v, text in refs or []:
        out.append(f'<line class="ref" x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{mt}" y2="{h - mb}"/>'
                   f'<text class="tick" x="{sx(v) + 4:.1f}" y="{mt + 9}">{escape(text)}</text>')
    for i, (name, v) in enumerate(zip(labels, values)):
        y = mt + row * i + 7
        x0, x1 = sorted((sx(0), sx(v)))
        bw = max(x1 - x0, 0.0)
        r = min(4.0, bw)
        cls = "s1" if highlight is None or i == highlight else "mute"
        if bw > 0.5:
            if v >= 0:
                d = f"M{x0:.1f},{y} H{x1 - r:.1f} Q{x1:.1f},{y} {x1:.1f},{y + r} V{y + 16 - r} Q{x1:.1f},{y + 16} {x1 - r:.1f},{y + 16} H{x0:.1f} Z"
            else:
                d = f"M{x1:.1f},{y} H{x0 + r:.1f} Q{x0:.1f},{y} {x0:.1f},{y + r} V{y + 16 - r} Q{x0:.1f},{y + 16} {x0 + r:.1f},{y + 16} H{x1:.1f} Z"
            out.append(f'<path class="{cls} fill" d="{d}"><title>{escape(name)}: {escape(fmt(v))}</title></path>')
        out.append(f'<text class="lbl" x="{label_w - 8}" y="{y + 12}" text-anchor="end">{escape(name)}</text>')
        tip = sx(v)
        if ci:
            a, b = sx(ci[i][0]), sx(ci[i][1])
            out.append(f'<line class="whisker" x1="{a:.1f}" x2="{b:.1f}" y1="{y + 8}" y2="{y + 8}"/>'
                       f'<line class="whisker" x1="{a:.1f}" x2="{a:.1f}" y1="{y + 4}" y2="{y + 12}"/>'
                       f'<line class="whisker" x1="{b:.1f}" x2="{b:.1f}" y1="{y + 4}" y2="{y + 12}"/>')
            tip = max(b, sx(0))
        out.append(f'<text class="val" x="{tip + 6:.1f}" y="{y + 12}">{escape(fmt(v))}</text>')
    out.append(f'<line class="axis" x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{mt}" y2="{h - mb}"/>')
    return _frame(w, h, "".join(out), label)


def calibration_chart(predicted: list[float], observed: list[float], label: str, w: int = 300, h: int = 280) -> str:
    """Observed against predicted event rate by score decile, log-log, with the y = x line."""
    ml, mr, mt, mb = 52, 14, 12, 40
    vals = [v for v in predicted + observed if v > 0]
    lo = 10 ** math.floor(math.log10(min(vals)))
    hi = 10 ** math.ceil(math.log10(max(vals)))
    s = lambda v, a, b: a + (b - a) * (math.log10(max(v, lo)) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))  # noqa: E731
    sx = lambda v: s(v, ml, w - mr)  # noqa: E731
    sy = lambda v: s(v, h - mb, mt)  # noqa: E731
    out, t = [], lo
    while t <= hi * 1.0001:
        txt = f"{t * 100:g}%"
        out.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                   f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{mt}" y2="{h - mb}"/>'
                   f'<text class="tick" x="{ml - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{txt}</text>'
                   f'<text class="tick" x="{sx(t):.1f}" y="{h - mb + 15}" text-anchor="middle">{txt}</text>')
        t *= 10
    out.append(f'<line class="ref" x1="{sx(lo):.1f}" y1="{sy(lo):.1f}" x2="{sx(hi):.1f}" y2="{sy(hi):.1f}"/>')
    for i, (p, o) in enumerate(zip(predicted, observed)):
        out.append(f'<circle class="s1 dot" cx="{sx(p):.1f}" cy="{sy(max(o, lo)):.1f}" r="4.5">'
                   f'<title>decile {i + 1}: predicted {p:.2%}, observed {o:.2%}</title></circle>')
    out.append(f'<text class="tick" x="{(ml + w - mr) / 2:.0f}" y="{h - 6}" text-anchor="middle">predicted rate</text>')
    out.append(f'<text class="tick" transform="translate(12 {(mt + h - mb) / 2:.0f}) rotate(-90)" text-anchor="middle">observed rate</text>')
    return _frame(w, h, "".join(out), label)


def step_chart(thresholds: list[float], values: list[float], lo: float, hi: float, title: str, missing: float,
               w: int = 300, h: int = 170) -> str:
    """One term of the additive model: contribution in log-odds against the feature value."""
    ml, mr, mt, mb = 44, 12, 24, 30
    if hi <= lo:
        hi = lo + 1
    xs = [lo] + [t for t in thresholds if lo < t < hi] + [hi]
    idx0 = sum(1 for t in thresholds if t <= lo)
    ys = values[idx0: idx0 + len(xs) - 1]
    y_lo, y_hi = min(ys + [missing, 0.0]), max(ys + [missing, 0.0])
    pad = (y_hi - y_lo) * 0.1 or 0.1
    y_lo, y_hi = y_lo - pad, y_hi + pad
    sx = lambda v: ml + (w - ml - mr) * (v - lo) / (hi - lo)  # noqa: E731
    sy = lambda v: mt + (h - mt - mb) * (1 - (v - y_lo) / (y_hi - y_lo))  # noqa: E731
    out = [f'<text class="lbl strong" x="{ml}" y="13">{escape(title)}</text>']
    for t in _ticks(y_lo, y_hi, 3):
        out.append(f'<line class="grid" x1="{ml}" x2="{w - mr}" y1="{sy(t):.1f}" y2="{sy(t):.1f}"/>'
                   f'<text class="tick" x="{ml - 6}" y="{sy(t) + 4:.1f}" text-anchor="end">{t:+g}</text>')
    out.append(f'<line class="axis" x1="{ml}" x2="{w - mr}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    pts = []
    for i, y in enumerate(ys):
        pts += [f"{sx(xs[i]):.1f},{sy(y):.1f}", f"{sx(xs[i + 1]):.1f},{sy(y):.1f}"]
    out.append(f'<polyline class="s1 line" points="{" ".join(pts)}"><title>{escape(title)}</title></polyline>')
    for v in (lo, hi):
        out.append(f'<text class="tick" x="{sx(v):.1f}" y="{h - mb + 14}" text-anchor="{"start" if v == lo else "end"}">{v:.3g}</text>')
    out.append(f'<text class="tick" x="{(ml + w - mr) / 2:.0f}" y="{h - 4}" text-anchor="middle">missing: {missing:+.2f}</text>')
    return _frame(w, h, "".join(out), f"shape of {title}")
