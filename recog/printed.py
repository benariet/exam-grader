"""The printed part of an exam booklet's pages - the same in every student's copy (question text, the header and logo,
frames, the footer, a printed graph grid) - found from many students' scans of it, so that what a student wrote can be
told from what was printed:

    template = build_template([ink of page k in exam 1, in exam 2, ...])     # ink: cleaned page < 128
    printed = printed_on(template, ink)                  # the template laid on one scan: its printed pixels there

Scans of one booklet differ a little: a shift, a slight turn, a little scale. Each scan is matched to the first one tile
by tile (phase correlation over a 4 x 4 grid) and an affine map fitted to the tiles' shifts (tiles that disagree with
the rest - mostly handwriting, nothing printed - left out); the template is what is ink in at least half of the
scans so laid on each other."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

TILES = (4, 4)
MAX_SHIFT = 60          # pixels at 150 dpi: a shift beyond this between two scans is a wrong match


def _tile_shifts(ref: np.ndarray, img: np.ndarray):
    """[(centre (row, col) in img, shift (row, col) onto ref, weight)] per tile with ink in both"""
    from skimage.registration import phase_cross_correlation
    H, W = min(ref.shape[0], img.shape[0]), min(ref.shape[1], img.shape[1])
    th, tw = H // TILES[0], W // TILES[1]
    out = []
    for i in range(TILES[0]):
        for j in range(TILES[1]):
            a = ref[i * th:(i + 1) * th, j * tw:(j + 1) * tw].astype(np.float32)
            b = img[i * th:(i + 1) * th, j * tw:(j + 1) * tw].astype(np.float32)
            if a.mean() < 0.004 or b.mean() < 0.004:
                continue
            a, b = ndi.gaussian_filter(a, 1.5), ndi.gaussian_filter(b, 1.5)
            try:
                shift, err, _ = phase_cross_correlation(a, b, upsample_factor=2, normalization=None)
            except Exception:                              # noqa: BLE001
                continue
            if np.abs(shift).max() > MAX_SHIFT:
                continue
            out.append(((i + 0.5) * th, (j + 0.5) * tw, float(shift[0]), float(shift[1]), min(a.mean(), b.mean())))
    return out


def fit_map(ref: np.ndarray, img: np.ndarray):
    """an affine map from img's (row, col) to ref's -> (M 2x2, b 2): p_ref = M p_img + b"""
    t = _tile_shifts(ref, img)
    if not t:
        return np.eye(2), np.zeros(2)
    t = np.array(t)
    med = np.median(t[:, 2:4], axis=0)
    ok = np.abs(t[:, 2:4] - med).max(1) <= 12            # (tiles agreeing with most: the printed ones)
    t = t[ok] if ok.sum() >= 1 else t
    if len(t) < 4:
        return np.eye(2), np.median(t[:, 2:4], axis=0)
    P = np.c_[t[:, 0], t[:, 1], np.ones(len(t))]
    Q = t[:, 0:2] + t[:, 2:4]
    w = np.sqrt(t[:, 4])
    sol, *_ = np.linalg.lstsq(P * w[:, None], Q * w[:, None], rcond=None)
    M, b = sol[:2].T, sol[2]
    if np.abs(M - np.eye(2)).max() > 0.03:               # (more turn or scale than a scanner gives: a bad fit)
        return np.eye(2), med
    return M, b


def to_ref(img: np.ndarray, M, b, shape) -> np.ndarray:
    """img laid in ref's frame"""
    Mi = np.linalg.inv(M)
    return ndi.affine_transform(img.astype(np.uint8), Mi, offset=-Mi @ b, output_shape=shape, order=0) > 0


def from_ref(ref_like: np.ndarray, M, b, shape) -> np.ndarray:
    """a picture in ref's frame laid on img's frame"""
    return ndi.affine_transform(ref_like.astype(np.uint8), M, offset=b, output_shape=shape, order=0) > 0


def build_template(inks: list, share: float = 0.5) -> np.ndarray:
    """the pixels that are ink in at least `share` of the scans (laid on the first one's frame), grown by a pixel"""
    ref = inks[0]
    acc = ref.astype(np.int16).copy()
    for ink in inks[1:]:
        M, b = fit_map(ref, ink)
        acc += to_ref(ink, M, b, ref.shape)
    return ndi.binary_dilation(acc >= max(2, share * len(inks)), iterations=1)


def printed_on(template: np.ndarray, ink: np.ndarray, grow: int = 2) -> np.ndarray:
    """the template laid on this scan (grown by `grow` pixels for what the match misses): its printed pixels"""
    M, b = fit_map(template, ink)
    p = from_ref(template, M, b, ink.shape)
    return ndi.binary_dilation(p, iterations=grow) if grow else p
