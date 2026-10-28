"""Handwriting recognizers. Each returns an n-best list of candidates with total log-probs,
which grader.Reading consumes (see readings_from_candidates)."""
import math
import os


def session_options():
    """onnxruntime SessionOptions shared by all recognizers. EXAM_GRADER_THREADS=n limits each model
    to n CPU threads (useful on a shared server, or while a training job is running)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    n = int(os.environ.get("EXAM_GRADER_THREADS", "0") or 0)
    if n > 0:
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
    return so


def readings_from_candidates(cands):
    """n-best candidates (text, logp) -> grader.Reading list with normalised probabilities."""
    from grader import Reading
    m = max(c.logp for c in cands)
    w = [math.exp(c.logp - m) for c in cands]
    z = sum(w)
    return [Reading(c.text, p / z) for c, p in zip(cands, w)]
