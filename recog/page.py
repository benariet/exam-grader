"""Full scanned pages (notebook / exam sheets) -> clean line images for MixedReader.

The answer-box pipeline assumes a clean background and blank space between lines. A real page has
ruled lines, a margin line, scanner borders and specks, and handwritten lines whose ascenders and
descenders touch the next line. Here:

  flatten_photo: a phone photo first: the paper is found (bright, grey - the table around it is not),
                straightened by a perspective warp, its light evened out (each pixel divided by the
                local paper brightness) and the notebook rules removed (long thin horizontal lines;
                a pen stroke that crosses a rule keeps its dark core). looks_like_photo() decides.
  clean_scan  : everything lighter than the gap between pen ink and rules/paper -> white; scanner
                borders, long vertical lines (margins), punch holes and specks removed; frames drawn
                around answers removed and their boxes kept (info["answer_boxes"])
  find_lines  : bands of rows with ink; a band taller than a line is split at deep valleys of the
                row profile (never right next to a long horizontal stroke: fraction bars and
                underlines stay with their line); then every connected stroke goes to the line
                holding its centre, so a descender that reaches into the next band stays with its
                own letter. Strokes much larger than a letter in both directions (a drawn graph,
                a signature) are returned as figures, not read as text.
"""
from __future__ import annotations

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


def ink_threshold(gray: np.ndarray) -> int:
    """Gray level between pen ink and rules / paper: the emptiest histogram bin in 130..200."""
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    smooth = np.convolve(hist, np.ones(9) / 9, mode="same")
    return int(130 + np.argmin(smooth[130:201]))


COLOR_INK = 80                  # max - min channel above this: a coloured pen, not paper or a light rule


def ink_gray(img: Image.Image) -> np.ndarray:
    """Grey levels in which coloured pens are dark: red ink (255, 139, 137) is 173 in plain luminance -
    as light as paper to the ink threshold. A strongly coloured pixel gets 255 - 1.5 x its saturation
    (red ink -> ~80, blue ink -> ~30) if that is darker; light blue notebook rules are not that
    saturated and stay as they are."""
    if img.mode == "L":
        return np.asarray(img, dtype=np.float32)
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    g = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    sat = rgb.max(2) - rgb.min(2)
    return np.where(sat > COLOR_INK, np.minimum(g, 255 - 1.5 * sat), g).astype(np.float32)


def clean_scan(img: Image.Image, min_speck: int = 12, answer_boxes: bool = True):
    """-> (grayscale page: white background, ink < 128; info dict)"""
    g = ink_gray(img)
    t = ink_threshold(g)
    page = np.where(g >= t, 255, g * (110.0 / t)).astype(np.uint8)
    ink = page < 128
    H, W = ink.shape
    m = max(2, int(0.012 * max(H, W)))                       # scanner borders and binding holes
    ink[:, :m] = ink[:, -m:] = False
    ink[-m:, :] = False
    top = 0
    for y in range(int(0.06 * H)):                           # spiral holes: a dotted band at the top
        if ink[y].mean() > 0.12:
            top = y + 1
    ink[: top + 4 if top else m] = False
    vert = ndi.binary_opening(ink, structure=np.ones((max(20, H // 40), 1)))   # long thin vertical lines
    cols = np.where(vert.sum(0) > 0.15 * H)[0]
    if len(cols):
        lines = np.zeros_like(ink)
        lines[:, cols] = vert[:, cols]
        ink &= ~ndi.binary_dilation(lines, structure=np.ones((3, 5)))
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    holes = []
    if n:
        sizes = ndi.sum(ink, lab, range(1, n + 1))
        small = np.where(sizes < min_speck)[0] + 1
        if len(small):
            # a speck far from any writing is dirt; one next to it may be a decimal point (26.67), a
            # dot over an i, a dot product - those stay
            lh0 = letter_height(ink)
            dist = ndi.distance_transform_edt(~(ink & ~np.isin(lab, small)))
            near = ndi.minimum(dist, lab, small) <= 0.8 * lh0
            ink &= ~np.isin(lab, small[~np.asarray(near)])
        for k, o in enumerate(ndi.find_objects(lab), 1):        # punch holes: filled round blobs at a side
            h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
            cx = (o[1].start + o[1].stop) / 2
            if 12 <= max(h, w) <= 0.045 * W and 0.6 < w / h < 1.6 and sizes[k - 1] > 0.55 * h * w \
                    and (cx < 0.1 * W or cx > 0.9 * W):
                holes.append(k)
        if holes:
            ink &= ~np.isin(lab, holes)
    boxes = []
    if answer_boxes:                 # a frame drawn around an answer: removed, its place remembered
        boxes, frames = find_answer_boxes(ink, letter_height(ink))
        ink &= ~frames
    page = np.where(ink, page, 255).astype(np.uint8)
    return Image.fromarray(page), {"ink_threshold": t, "top_cut": top, "margin_line_columns": len(cols),
                                   "punch_holes": len(holes), "answer_boxes": boxes}


def find_answer_boxes(ink: np.ndarray, lh: float):
    """Hand-drawn frames around answers (students box their final result): a stroke whose ink runs
    along all four sides of its bounding box and that holds writing inside. A frame joins the line
    above and the line below into one and the readers read it as letters - so it is removed; its
    box is returned, because what it holds is most likely the final answer.
    -> ([(x0, y0, x1, y1)], mask of the frame pixels)"""
    frames = np.zeros_like(ink)
    boxes = []
    t = max(3, int(round(0.25 * lh)))                    # how far in from the outline a frame stroke reaches
    # a frame drawn in one stroke is one component; one drawn in four strokes often has open corners:
    # a second pass looks at strokes joined across small gaps
    gap = max(2, int(round(0.3 * lh)))
    for joined in (ink, ndi.binary_dilation(ink, iterations=gap)):
        for o, m in _frame_candidates(joined, ink & ~frames, lh):
            h, w = m.shape
            box = _frame(m, ink, o, h, w, t, lh)
            if box is not None:
                frames[o] |= box
                boxes.append((o[1].start, o[0].start, o[1].stop, o[0].stop))
    return boxes, frames


def _frame_candidates(joined, ink, lh):
    lab, n = ndi.label(joined, structure=np.ones((3, 3)))
    for k, o in enumerate(ndi.find_objects(lab), 1):
        h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
        if 1.2 * lh <= h <= 6.5 * lh and w >= 2.0 * lh and w >= 1.2 * h:
            m = (lab[o] == k) & ink[o]
            if m.any():
                yield o, m


def _frame(m, ink, o, h, w, t, lh):
    """the frame pixels of candidate m (its bounding box o), or None if it is not a frame"""
    ys, xs = np.nonzero(m)
    top = np.full(w, h); bot = np.full(w, -1); left = np.full(h, w); right = np.full(h, -1)
    np.minimum.at(top, xs, ys); np.maximum.at(bot, xs, ys)
    np.minimum.at(left, ys, xs); np.maximum.at(right, ys, xs)
    bx, by = 0.3 * h, 0.15 * w                        # a side is the outline within this of the box edge
    cx = slice(int(0.1 * w), max(int(0.1 * w) + 1, int(0.9 * w)))
    cy = slice(int(0.2 * h), max(int(0.2 * h) + 1, int(0.8 * h)))
    sides = [(top[cx] <= bx).mean(), (bot[cx] >= h - 1 - bx).mean(),
             (left[cy] <= by).mean(), (right[cy] >= w - 1 - by).mean()]
    if min(sides) < 0.8:                  # writing has gaps between letters: no ink at the top
        return None                       # and the bottom of 80 % of its columns
    yy, xx = np.mgrid[0:h, 0:w]
    on_frame = m & ((yy <= top[None, :] + t) | (yy >= bot[None, :] - t) |
                    (xx <= left[:, None] + t) | (xx >= right[:, None] - t))
    if on_frame.sum() < 0.3 * m.sum():    # mostly outline (writing touching the frame is allowed)
        return None
    inner = ink[o[0].start + t: o[0].stop - t, o[1].start + t: o[1].stop - t] & ~on_frame[t:h - t, t:w - t]
    if inner.sum() < 2 * lh:              # an empty rectangle is not an answer box
        return None
    return on_frame


# ------------------------------------------------------------------ phone photos
def _paper_mask(img: Image.Image, side: int = 800):
    """-> (mask of the paper at low resolution, scale): bright and grey (a wooden or coloured table
    is more saturated, a shadow is dark); the largest such region"""
    scale = side / max(img.size)
    hsv = np.asarray(img.convert("RGB").resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))),
                                               Image.BILINEAR).convert("HSV"))
    s, v = hsv[..., 1].astype(np.float32), hsv[..., 2].astype(np.float32)
    hist = np.histogram(s, bins=256, range=(0, 256))[0].astype(np.float64)
    ts = _otsu(hist)
    ts = min(max(ts, 25), 60)
    low = s < ts
    tv = 0.5 * np.median(v[low]) if low.any() else 128
    mask = low & (v > tv)
    mask = ndi.binary_closing(mask, structure=np.ones((9, 9)))
    mask = ndi.binary_opening(mask, structure=np.ones((9, 9)))
    lab, n = ndi.label(mask)
    if not n:
        return None, scale
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes))), scale


def _otsu(hist: np.ndarray) -> int:
    p = hist / max(1.0, hist.sum())
    omega, mu = np.cumsum(p), np.cumsum(p * np.arange(len(p)))
    between = (mu[-1] * omega - mu) ** 2 / np.maximum(omega * (1 - omega), 1e-12)
    return int(np.argmax(between))


def looks_like_photo(img: Image.Image) -> bool:
    """A scan is paper filling the frame under even light; a photo shows the table around the page,
    or light that falls off across it."""
    mask, scale = _paper_mask(img)
    if mask is None:
        return False
    if mask.mean() < 0.9:
        return True
    v = np.asarray(img.convert("L").resize(mask.shape[::-1], Image.BILINEAR), dtype=np.float32)
    v = ndi.gaussian_filter(v, 8)[mask]
    return float(np.percentile(v, 95) - np.percentile(v, 5)) > 40


def _corners(mask: np.ndarray):
    """paper corners (TL, TR, BR, BL) as (x, y): the extreme points along the two diagonals"""
    ys, xs = np.nonzero(mask)
    s, d = xs + ys, xs - ys
    return np.array([[xs[np.argmin(s)], ys[np.argmin(s)]], [xs[np.argmax(d)], ys[np.argmax(d)]],
                     [xs[np.argmax(s)], ys[np.argmax(s)]], [xs[np.argmin(d)], ys[np.argmin(d)]]], np.float64)


def _warp(img: Image.Image, quad: np.ndarray, width: int) -> Image.Image:
    """perspective-warp the quadrilateral (TL, TR, BR, BL) to an upright page `width` pixels wide"""
    tl, tr, br, bl = quad
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    W, H = width, max(1, int(round(width * h / max(w, 1))))
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float64)
    A, b = [], []
    for (x, y), (u, v) in zip(dst, quad):             # output pixel (x, y) <- input (u, v)
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.append(u)
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y]); b.append(v)
    coeffs = np.linalg.solve(np.array(A), np.array(b))
    return img.transform((W, H), Image.PERSPECTIVE, tuple(coeffs), Image.BICUBIC)


def _even_light(gray: np.ndarray) -> np.ndarray:
    """divide by the local paper brightness (ink removed by a max filter): paper -> white everywhere"""
    k = max(15, gray.shape[1] // 60)
    bg = ndi.gaussian_filter(ndi.maximum_filter(gray, size=k), k * 0.6)
    return np.clip(gray / np.maximum(bg, 1) * 255, 0, 255)


def _long_runs(S: np.ndarray, L: int, gap: int) -> np.ndarray:
    """per row: the runs of True (breaks up to `gap` bridged) at least L long, breaks filled"""
    H, W = S.shape
    p = np.zeros((H, W + 2), np.int8)
    p[:, 1:-1] = S
    d = np.diff(p, axis=1)
    r, s = np.nonzero(d == 1)
    _, e = np.nonzero(d == -1)
    if not len(r):
        return np.zeros_like(S)
    brk = np.ones(len(r), bool)
    brk[1:] = (r[1:] != r[:-1]) | (s[1:] - e[:-1] > gap)
    first = np.nonzero(brk)[0]
    last = np.r_[first[1:] - 1, len(r) - 1]
    gs, ge, gr = s[first], e[last], r[first]
    ok = ge - gs >= L
    acc = np.zeros((H, W + 1), np.int32)
    np.add.at(acc, (gr[ok], gs[ok]), 1)
    np.add.at(acc, (gr[ok], ge[ok]), -1)
    return np.cumsum(acc, axis=1)[:, :W] > 0


def _line_opening(m: np.ndarray, t: float, L: int, gap: int) -> np.ndarray:
    """the pixels of m on straight runs of slope t (rows per column) at least L long: a horizontal opening of the
    page sheared by t (a row either way tolerated, breaks up to `gap` bridged)"""
    H, W = m.shape
    xs = np.arange(W)
    sh = np.round(t * xs).astype(int)
    sh -= sh.min()
    P = int(sh.max())
    mp = np.zeros((H + 2 * P + 2, W), bool)
    mp[P + 1: P + 1 + H] = m
    mp[1:-1] |= mp[:-2] | mp[2:]
    K = _long_runs(mp[np.arange(H + P)[:, None] - sh[None, :] + P + 1, xs[None, :]], L, gap)
    return K[np.arange(H)[:, None] + sh[None, :], xs[None, :]]


def tilted_lines(dark: np.ndarray, L: int, max_slope: float = 0.15, gap: int = 25) -> np.ndarray:
    """long thin straight lines at any slope up to max_slope (a photo's rules are seldom level, and bend a little):
    the union of sheared openings, found at half resolution"""
    H, W = dark.shape
    Hs, Ws = H // 2, W // 2
    small = dark[: Hs * 2, : Ws * 2].reshape(Hs, 2, Ws, 2).any(axis=(1, 3))
    Ls = max(10, L // 2)
    found = np.zeros_like(small)
    for t in np.arange(-max_slope, max_slope + 1e-9, 2.0 / Ls):      # a row's drift over a line's length
        found |= _line_opening(small, t, Ls, max(1, gap // 2))
    up = np.zeros((H, W), bool)
    up[: Hs * 2, : Ws * 2] = np.repeat(np.repeat(found, 2, 0), 2, 1)
    return up


def rule_lines(norm: np.ndarray, max_slope: float = 0.0):
    """the long thin horizontal lines of a page (a short break bridged, a slight tilt tolerated - any slope up to
    max_slope when given) - notebook rules, or a drawing's long straight strokes -> (mask, letter height)"""
    H, W = norm.shape
    lh = letter_height(norm < 140)
    dark = norm < 205
    L = int(max(8 * lh, 0.2 * W))
    if max_slope:
        return tilted_lines(dark, L, max_slope), lh
    x = ndi.binary_dilation(dark, structure=np.ones((3, 1)))
    x = ndi.binary_closing(x, structure=np.ones((1, 25)))
    return ndi.binary_opening(x, structure=np.ones((1, L))), lh


def _remove_rules(norm: np.ndarray, which=None, max_slope: float = 0.0):
    """Notebook rules -> white. Rules: long thin horizontal lines (a short break is bridged, a slight
    tilt tolerated). On a rule only the stroke cores clearly darker than the rule *at that place*
    stay (a rule is not equally dark everywhere in a photo), and what is left lying wholly on a rule
    (a darker bit of the rule itself) goes too. which: a function of (the lines' mask, the page) -> the part of
    the mask that is rules (default: all of it). max_slope: rules at any slope up to this (rule_lines); the rule's
    level at a place is then taken from its lighter pixels there and never under the rule's own shade over its
    whole length - a wire drawn along or right on a rule is no outlier then - and a leftover clearly darker than
    the rule there, or long, is the pen's and stays. -> (page, median rule level)"""
    H, W = norm.shape
    hor, lh = rule_lines(norm, max_slope)
    dark = norm < 205
    if which is not None and hor.any():
        hor = which(hor, norm)
    if not hor.any():
        return norm, None
    band = hor & ndi.binary_dilation(dark, structure=np.ones((3, 1)))   # the rule's own rows (+-1 px)
    if not band.any():
        return norm, None
    rule_level = float(np.median(norm[band & dark]))
    keep = np.zeros_like(band)
    level = np.full(norm.shape, 255.0)                         # the rule's level at each of its pixels
    win = max(31, int(3 * lh) | 1)
    if max_slope:                  # each rule's own shade, over its whole length: its darkest pixel in each column,
        rl, rn = ndi.label(hor, structure=np.ones((3, 3)))     # the paler quarter's (where the pen runs on it, the
        own_of = np.zeros(rn + 1)                              # rule is not that dark)
        for k, o in enumerate(ndi.find_objects(rl), 1):
            sel = (rl[o] == k) & dark[o]
            if o is not None and sel.any():
                own_of[k] = np.percentile(np.where(sel, norm[o], 255.0).min(0)[sel.any(0)], 75)
        own_map = own_of[rl]
    lab, n = ndi.label(band, structure=np.ones((3, 3)))
    for k, o in enumerate(ndi.find_objects(lab), 1):          # one band per rule (a faint stretch may split it)
        sel = lab[o] == k
        sub = np.where(sel, norm[o], 255.0)
        if max_slope:              # the rule's level in each column: its lighter dark pixels (a wire running beside
            ink = sel & (norm[o] < 205)       # the rule, joined to it, is darker), crossings smoothed - and never
            cols = np.nonzero(ink.any(0))[0]  # under the rule's own shade: a wire drawn right on it stays
            local = np.full(sub.shape[1], 255.0)
            if len(cols):
                with np.errstate(all="ignore"):
                    lv = np.nanpercentile(np.where(ink, norm[o], np.nan)[:, cols], 80, axis=0)
                own = float(np.median(own_map[o][sel]))
                local[cols] = np.maximum(ndi.median_filter(lv, size=win, mode="nearest"), own)
        else:
            darkest = sub.min(0)                                # the rule's darkness in each column
            local = ndi.median_filter(darkest, size=win, mode="nearest")   # crossing strokes are outliers
        keep[o] |= sel & (sub < local[None, :] - 25)
        if max_slope:
            level[o] = np.where(sel, local[None, :], level[o])
    out = norm.copy()
    out[band & ~keep] = 255
    # a stroke that crosses the rule: ink right above and right below the removed pixels -> put them back
    above = np.zeros_like(band); below = np.zeros_like(band)
    for d in (2, 3, 4):
        above[d:] |= norm[:-d] < 175
        below[:-d] |= norm[d:] < 175
    back = band & ~keep & above & below
    out[back] = norm[back]
    band = ndi.binary_dilation(band, structure=np.ones((5, 1)))       # wider, for the leftovers test
    ink = out < 200
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    if n:
        outside = np.asarray(ndi.sum(ink & ~band, lab, range(1, n + 1)))
        gone = outside == 0
        if max_slope:     # a stroke drawn along a rule, much darker than the rule there - or clearly darker and
            on = ndi.binary_dilation(level < 255, iterations=3)               # long - is the pen's, not a darker
            lv = np.where(on, ndi.grey_erosion(level, size=7), 255.0)          # bit of the rule
            under = np.asarray(ndi.median(lv - out, lab, range(1, n + 1)))
            span = np.array([o[1].stop - o[1].start if o is not None else 0 for o in ndi.find_objects(lab)])
            gone &= (under < 25) | ((under < 45) & (span < lh))
        out[np.isin(lab, np.where(gone)[0] + 1)] = 255
    return out, rule_level


def flatten_photo(img: Image.Image, width: int = 1400):
    """phone photo -> (grayscale page as from a scanner, info)"""
    mask, scale = _paper_mask(img)
    info = {"photo": True}
    if mask is not None and mask.mean() > 0.2:
        quad = _corners(mask) / scale
        info.update(paper_share=round(float(mask.mean()), 3), corners=quad.round().astype(int).tolist())
        page = _warp(img.convert("RGB"), quad, width)
    else:
        page = img.convert("RGB").resize((width, max(1, round(img.height * width / img.width))), Image.BICUBIC)
    norm = _even_light(np.asarray(page.convert("L"), dtype=np.float32))
    norm, rule_level = _remove_rules(norm)
    info["rule_level"] = rule_level
    return Image.fromarray(norm.astype(np.uint8)), info


def letter_height(ink: np.ndarray) -> float:
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    if not n:
        return 30.0
    objs = ndi.find_objects(lab)
    hs = np.array([o[0].stop - o[0].start for o in objs])
    areas = ndi.sum(ink, lab, range(1, n + 1))
    ok = (areas > 30) & (hs < 0.08 * ink.shape[0])
    return float(np.median(hs[ok])) if ok.any() else 30.0


def _bands(rows: np.ndarray, min_gap: int = 6):
    idx = np.where(rows)[0]
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > min_gap:
            out.append([int(s), int(p) + 1]); s = i
        p = i
    out.append([int(s), int(p) + 1])
    return out


def _longest_runs(ink: np.ndarray) -> np.ndarray:
    """per row: length of the longest horizontal run of ink"""
    out = np.zeros(ink.shape[0], int)
    for y in np.where(ink.any(1))[0]:
        row = np.concatenate([[0], ink[y].astype(np.int8), [0]])
        d = np.diff(row)
        starts, ends = np.where(d == 1)[0], np.where(d == -1)[0]
        out[y] = (ends - starts).max()
    return out


def _split(profile: np.ndarray, bars: np.ndarray, a: int, b: int, lh: float, depth: float = 0.35, keep=()):
    """Split rows a..b at the deepest valley that is much lower than the peaks on both sides,
    if both parts are at least one letter tall and no long horizontal stroke (fraction bar,
    underline) is within about a letter of the cut; recurse."""
    if b - a < 2.2 * lh:
        return [(a, b)]
    w, near = max(3, int(lh // 2)), max(3, int(1.0 * lh))
    best = None
    for y in range(a + int(lh), b - int(lh)):
        left, right = profile[max(a, y - 2 * int(lh)): y].max(), profile[y + 1: min(b, y + 2 * int(lh))].max()
        if profile[y] == profile[max(a, y - w): y + w + 1].min() and profile[y] < depth * min(left, right) \
                and bars[max(a, y - near): min(b, y + near)].max(initial=0) < 1.5 * lh \
                and not any(t < y < bot for t, bot in keep):           # never inside a fraction
            score = profile[y] / max(1e-6, min(left, right))
            if best is None or score < best[0]:
                best = (score, y)
    if best is None:
        return [(a, b)]
    y = best[1]
    return _split(profile, bars, a, y, lh, depth, keep) + _split(profile, bars, y, b, lh, depth, keep)


def _band_letter_height(ink, a, b, lh):
    """the letter height of the writing in rows a..b (the page's median can be far smaller on a page
    full of small print or drawing labels) - never below the page's"""
    lab, n = ndi.label(ink[a:b], structure=np.ones((3, 3)))
    if not n:
        return lh
    hs = np.array([o[0].stop - o[0].start for o in ndi.find_objects(lab)])
    ok = np.asarray(ndi.sum(ink[a:b], lab, range(1, n + 1))) >= 20
    return max(lh, float(np.median(hs[ok]))) if ok.any() else lh


SHORT_MIN_W = 2.0           # a short line squeezed under a long one: at least this x the letter height wide,
SHORT_EDGE = 1.5            # ... starting where the long line starts (right or left, within this x the letter height),
SHORT_BAR = 0.8             # ... no long thin stroke this long (a fraction bar, a minus, a wire) near the cut, in its columns,
SHORT_COVER = 0.5           # ... and letters of the long line over this share of it (gaps between letters bridged)


def _letters(ink, y0, y1, x0, x1, lh):
    """letter-like strokes in rows y0..y1, columns x0..x1: (top, bottom, left, right) of every stroke at
    least 0.4 letter heights tall and at most 2 wide (a long curve, a wire or a frame is no letter)"""
    lab, n = ndi.label(ink[y0:y1, x0:x1], structure=np.ones((3, 3)))
    out = []
    for o in ndi.find_objects(lab):
        h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
        if h >= 0.4 * lh and w <= 2 * lh:
            out.append((y0 + o[0].start, y0 + o[0].stop, x0 + o[1].start, x0 + o[1].stop))
    return out


def _split_local(ink, bars, a, b, lh, keep=()):
    """A short line written right under a long one - the end of a sentence squeezed onto the next
    notebook line - leaves no valley in the profile of the whole width: the long line's descenders
    fill it, and the short line's own peak is small. Look again with the profile of the short line's
    columns only (the columns of the strokes that start in the band's lower half, widened by a quarter
    letter), with the same valley rule as _split. What tells such a line from the parts of a formula
    or a drawing (a subscript, a denominator, the lower end of a slanted line, a label under a wire):
    it is a few letters wide, starts where the long line starts (right or left), letters of the long
    line run over it, there is no fraction bar or wire near the cut, and it has three or more letters
    of its own below the cut, one after the other like words. Only below: over a line, the same
    picture is far more often an exponent, numerators or the rising end of a slanted line (measured
    on real exam pages). -> [(top, bottom, None or (the short line's columns, its left, its right))]
    top to bottom"""
    lh = _band_letter_height(ink, a, b, lh)
    if b - a < 2.2 * lh:
        return [(a, b, None)]
    h = max(3, int(lh))
    band_cols = ink[a:b].any(0)
    xs = np.where(band_cols)[0]
    lab, _ = ndi.label(ink[a:b], structure=np.ones((3, 3)))        # the short line's columns: under strokes
    cols = np.zeros(ink.shape[1], bool)                             # that start in the band's lower half
    for o in ndi.find_objects(lab):                                 # (a descender of the long line starts
        if a + o[0].start >= (a + b) / 2 and o[0].stop - o[0].start >= 0.3 * lh:   # higher up)
            cols[o[1].start:o[1].stop] = True
    if not cols.any():
        return [(a, b, None)]
    cols = ndi.binary_dilation(cols, iterations=max(1, h // 4)) & band_cols
    if cols.sum() >= 0.8 * band_cols.sum():                 # (almost) the whole width: _split saw this already
        return [(a, b, None)]
    cx = np.where(cols)[0]
    x0, x1 = int(cx[0]), int(cx[-1]) + 1
    if x1 - x0 < SHORT_MIN_W * lh or min(x0 - xs[0], xs[-1] - x1) > SHORT_EDGE * lh:
        return [(a, b, None)]
    prof = np.convolve(ink[:, cols].sum(1).astype(float), np.ones(7) / 7, mode="same")
    parts = _split(prof, bars, a, b, lh, keep=keep)
    if len(parts) < 2:
        return [(a, b, None)]
    y = parts[-1][0]                                        # the cut right over the short line
    y0n, y1n = max(a, y - h), min(b, y + h)                 # a fraction bar, a minus, a wire near the cut:
    lab_n, _ = ndi.label(ink[y0n:y1n] & cols[None, :], structure=np.ones((3, 3)))     # a long thin stroke
    for o in ndi.find_objects(lab_n):
        bw, bh = o[1].stop - o[1].start, o[0].stop - o[0].start
        if bw >= SHORT_BAR * lh and bw >= 3 * bh:
            return [(a, b, None)]
    over = np.zeros(x1 - x0, bool)                          # the long line's letters over the short line
    for t, s, l, r in _letters(ink, max(a, y - int(1.5 * lh)), y, x0, x1, lh):
        over[max(0, l - x0 - h // 2): r - x0 + h // 2] = True
    under = [c for c in _letters(ink, a, b, x0, x1, lh) if c[0] >= y - h // 4]    # letters starting below the cut
    if over.mean() < SHORT_COVER or len(under) < 3:
        return [(a, b, None)]
    reach = None                                            # ... one after the other, like the words of a line
    for l, r in sorted((l, r) for _, _, l, r in under):
        if reach is not None and l - reach > 2 * lh:
            return [(a, b, None)]                           # labels scattered under a drawing
        reach = r if reach is None else max(reach, r)
    return _split_local(ink, bars, a, y, lh, keep) + [(y, b, (cols, x0, x1))]


def _bar_components(lab, lh):
    """long thin horizontal strokes: (label, x0, y0, x1, y1)"""
    out = []
    for k, o in enumerate(ndi.find_objects(lab), 1):
        h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
        if w >= 2.0 * lh and h <= 0.6 * lh and w >= 4 * h:
            out.append((k, o[1].start, o[0].start, o[1].stop, o[0].stop))
    return out


def _bar_runs(ink, lh):
    """long, roughly horizontal strokes (fraction bars, underlines) - also slanted ones and ones a
    letter touches: thicken vertically, keep what survives an opening with a long horizontal
    line, -> (None, x0, y0, x1, y1)"""
    dil = ndi.binary_dilation(ink, structure=np.ones((5, 1)))
    hor = ndi.binary_opening(dil, structure=np.ones((1, max(5, int(1.5 * lh)))))
    lab, _ = ndi.label(hor)
    out = []
    for o in ndi.find_objects(lab):
        h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
        if w >= 2 * lh and h <= 0.6 * lh + 6:
            out.append((None, o[1].start, o[0].start, o[1].stop, o[0].stop))
    return out


def _fraction_span(ink, bar, lh):
    """(top, bottom) of numerator..denominator if the bar has writing right above and right below
    it that stays within its width (a fraction), else None (an underline, a dash). The span grows
    over strokes within the bar's width that touch it vertically (exponents: x^2 over x)."""
    _, x0, y0, x1, y1 = bar
    H, W = ink.shape
    reach, m, wide = int(2.0 * lh), int(0.3 * lh), int(2 * lh)
    # the stroke that carries the bar must be a bar: a curve or a wave drawn between two text lines
    # has a straight stretch too, but the stroke goes on beyond it
    wy0, wy1, wx0, wx1 = max(0, y0 - reach), min(H, y1 + reach), max(0, x0 - 3 * int(lh)), min(W, x1 + 3 * int(lh))
    lab, n = ndi.label(ink[wy0:wy1, wx0:wx1], structure=np.ones((3, 3)))
    on_bar = lab[y0 - wy0: y1 - wy0, x0 - wx0: x1 - wx0]
    ks, counts = np.unique(on_bar[on_bar > 0], return_counts=True)
    if len(ks):
        o = ndi.find_objects(lab)[int(ks[np.argmax(counts)]) - 1]
        if o[1].start + wx0 < x0 - lh or o[1].stop + wx0 > x1 + lh:
            return None
    for a, b in ((max(0, y0 - reach), y0), (y1, min(H, y1 + reach))):
        inner = ink[a:b, max(0, x0 - m): x1 + m]
        outer = ink[a:b, max(0, x0 - wide): x1 + wide]
        if inner.sum() < 3 * lh or inner.sum() < 0.9 * outer.sum():
            return None
    rows = np.where(ink[:, max(0, x0 - m): x1 + m].any(1))[0]
    gap = max(2, int(0.5 * lh))
    above, below = rows[(rows < y0) & (rows >= y0 - reach)], rows[(rows >= y1) & (rows < y1 + reach)]
    top, bot = int(above.max()), int(below.min()) + 1       # nearest writing above / below the bar,
    for y in rows[rows < top][::-1]:                         # then grow over small gaps (exponents)
        if top - y > gap:
            break
        top = int(y)
    for y in rows[rows >= bot]:
        if y - bot >= gap:
            break
        bot = int(y) + 1
    return top, bot


def _separators(lab, lh):
    """long, thin, mostly empty strokes running across the page - a line or a wave drawn between
    answers: (label, x0, y0, x1, y1). Underlines are handled separately (they can be short)."""
    out = []
    for k, o in enumerate(ndi.find_objects(lab), 1):
        h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
        if w >= 8 * lh and h <= 2.5 * lh and w >= 5 * h:
            cols = (lab[o] == k).sum(0)
            if cols[cols > 0].mean() <= 0.3 * lh:             # one stroke thick all along: no letters
                out.append((k, o[1].start, o[0].start, o[1].stop, o[0].stop))
    return out


def find_figures(ink: np.ndarray, lh: float):
    """-> (labels, n, figure label ids): connected strokes much larger than a letter in both
    directions and mostly empty inside (graphs, sketches, signatures)."""
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    figs = []
    for k, o in enumerate(ndi.find_objects(lab), 1):
        h, w = o[0].stop - o[0].start, o[1].stop - o[1].start
        if h > 3.5 * lh and w > 3.5 * lh and (lab[o] == k).mean() < 0.15:
            figs.append(k)
    return lab, n, figs


def find_lines(page: Image.Image, return_figures: bool = False):
    """-> [(y0, y1, line image of the full page width holding only that line's strokes)]
    (+ figure boxes [(x0, y0, x1, y1)] with return_figures=True)"""
    gray = np.asarray(page.convert("L"))
    ink = gray < 128
    lh = letter_height(ink)
    lab, n, figs = find_figures(ink, lh)
    fig_boxes = []
    objs = ndi.find_objects(lab)
    for k in figs:
        o = objs[k - 1]
        fig_boxes.append((o[1].start, o[0].start, o[1].stop, o[0].stop))
    if figs:
        ink = ink & ~np.isin(lab, figs)
        gray = np.where(ink, gray, 255).astype(np.uint8)
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    fractions = [sp for sp in (_fraction_span(ink, bar, lh) for bar in _bar_runs(ink, lh)) if sp]
    underlines = [bar[0] for bar in _bar_components(lab, lh)       # an underline carries no text
                  if bar[3] - bar[1] >= 3 * lh and not _fraction_span(ink, bar, lh)]
    underlines += [sep[0] for sep in _separators(lab, lh)           # a wave is never a fraction bar
                   if sep[4] - sep[2] > 1.0 * lh or not _fraction_span(ink, sep, lh)]
    if underlines:
        ink = ink & ~np.isin(lab, underlines)
        gray = np.where(ink, gray, 255).astype(np.uint8)
    rows = ink.sum(1) > 0
    profile = np.convolve(ink.sum(1).astype(float), np.ones(7) / 7, mode="same")
    bars = _longest_runs(ink)
    merged = []                       # numerator, bar and denominator may be separate bands: join
    for a, b in _bands(rows):         # every band that a fraction spans
        if merged and any(t < b and a < bot and t < merged[-1][1] and merged[-1][0] < bot for t, bot in fractions):
            merged[-1][1] = b
        else:
            merged.append([a, b])
    bands = []
    for a, b in merged:
        if b - a >= max(8, int(0.4 * lh)):
            for p, q in _split(profile, bars, a, b, lh, keep=fractions):
                bands += _split_local(ink, bars, p, q, lh, keep=fractions)
    if not bands:
        return ([], fig_boxes) if return_figures else []
    lab, n = ndi.label(ink, structure=np.ones((3, 3)))
    centers = ndi.center_of_mass(ink, lab, range(1, n + 1))
    mids = np.array([(a + b) / 2 for a, b, _ in bands])
    spans = [(o[0].start, o[0].stop) for o in ndi.find_objects(lab)]
    owner = np.zeros(n + 1, int) - 1
    reach = [0] * len(bands)
    for k, (cy, cx) in enumerate(centers, 1):
        inside = [i for i, (a, b, _) in enumerate(bands) if a <= cy < b]
        i = inside[0] if inside else int(np.argmin(np.abs(mids - cy)))
        a, b, short = bands[i]
        if short is not None and i > 0:      # in a short line's rows: its own, or a descender of the line above?
            cols = short[0]
            ll = _band_letter_height(ink, a, b, lh)
            top, bottom = spans[k - 1]
            if not cols[min(len(cols) - 1, int(cx))] and top < a - 0.15 * ll and bottom <= bands[i - 1][1] + 1.5 * ll:
                i -= 1
                reach[i] = max(reach[i], bottom)    # keep the whole descender in that line's picture
        owner[k] = i
    out = []
    for i, (a, b, _) in enumerate(bands):
        mask = owner[lab] == i
        # a long stray stroke (a big 'f', a slash) must not make the line much taller: the models
        # scale the whole line to a fixed height, and the writing would shrink to a smudge
        lo, hi = max(0, int(a - 0.6 * lh)), min(ink.shape[0], max(int(b + 0.6 * lh), reach[i]))
        mask[:lo] = False
        mask[hi:] = False
        ys = np.where(mask.any(1))[0]
        if not len(ys):
            continue
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        line = np.where(mask[y0:y1], gray[y0:y1], 255).astype(np.uint8)
        out.append((y0, y1, Image.fromarray(line)))
    return (out, fig_boxes) if return_figures else out
