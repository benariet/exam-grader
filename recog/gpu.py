"""Where a torch model runs and in which number format.

* a GPU from Ampere on (RTX 30xx / 40xx, A100 ...: compute capability 8 and up): bfloat16 - native, and
  as wide in range as float32.
* an older GPU (Turing - RTX 20xx such as the RTX 2080 Ti -, Volta, Pascal: 7.x and below) only emulates
  bfloat16, slowly. float16 is fast there, but its range ends at 65504 and some models overflow in it.
  So: float32 when the model fits in the GPU's free memory (a 2B reader: ~8 GB), else float16; a model
  asked to be fast (the language model, which scores hundreds of sentences per line) gets float16. A
  reader that overflows in float16 (NaN) is reloaded in float32 by its own class (`fallback_dtype`).
* CPU: bfloat16 where the processor computes it (AVX512-BF16 / AMX: recent Xeons) or memory is short,
  else float32 (an older Xeon such as the Gold 6130 has no bfloat16 instructions: float32 is faster).

EXAM_GRADER_DTYPE=fp32|fp16|bf16 overrides the format, EXAM_GRADER_DEVICE=cpu|cuda the device.
"""
from __future__ import annotations

import gc
import json
import os

import torch

NAMES = {"fp32": torch.float32, "float32": torch.float32, "fp16": torch.float16, "float16": torch.float16,
         "half": torch.float16, "bf16": torch.bfloat16, "bfloat16": torch.bfloat16}
SHORT = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}
MARGIN_GB = 0.8              # GPU memory left for activations, the KV cache and the display


def device(prefer: str | None = None) -> str:
    d = (prefer or os.environ.get("EXAM_GRADER_DEVICE", "")).strip().lower()
    if d.startswith("cuda") and torch.cuda.is_available():
        return d
    if d == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def params_b(model_dir: str) -> float | None:
    """billions of parameters of a downloaded model (from the size of its weight files and their format)"""
    try:
        files = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors")]
        size = sum(os.path.getsize(f) for f in files)
        cfg = json.load(open(os.path.join(model_dir, "config.json"), encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not size:
        return None
    fmt = cfg.get("torch_dtype") or cfg.get("dtype") or (cfg.get("text_config") or {}).get("dtype") or "bfloat16"
    return size / (4 if "32" in str(fmt) else 2) / 1e9


def gpu_free_gb() -> float:
    return torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0.0


def _fits_fp32(n_b: float | None) -> bool:
    return n_b is not None and n_b * 4e9 / 2**30 + MARGIN_GB <= gpu_free_gb()


def _cpu_bf16() -> bool:
    try:
        flags = open("/proc/cpuinfo", encoding="utf-8").read()
    except OSError:
        return True                          # not Linux: not known - bfloat16, as before
    return "avx512_bf16" in flags or "amx_bf16" in flags


def _ram_free_gb() -> float | None:
    """free memory of the machine (Linux: /proc/meminfo, Windows: GlobalMemoryStatusEx), None if unknown"""
    try:
        for line in open("/proc/meminfo", encoding="utf-8"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 2**20
    except OSError:
        pass
    if os.name == "nt":
        import ctypes

        class Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = Status()
        st.dwLength = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullAvailPhys / 2**30
    return None


def pick(n_b: float | None = None, dev: str | None = None, fast: bool = False) -> tuple[str, torch.dtype]:
    """-> (device, dtype) for a model of n_b billion parameters. fast: speed over range on an older GPU
    (float16 even when float32 would fit - the caller checks for overflow)"""
    dev = device(dev)
    env = os.environ.get("EXAM_GRADER_DTYPE", "").strip().lower()
    if env in NAMES:
        return dev, NAMES[env]
    if dev.startswith("cuda"):
        if torch.cuda.get_device_capability()[0] >= 8:
            return dev, torch.bfloat16
        return dev, (torch.float32 if not fast and _fits_fp32(n_b) else torch.float16)
    if _cpu_bf16():
        return dev, torch.bfloat16
    ram = _ram_free_gb()
    if n_b is not None and ram is not None and n_b * 4e9 / 2**30 * 1.3 > ram:
        return dev, torch.bfloat16
    return dev, torch.float32


def needs_4bit(n_b: float | None) -> bool:
    """a model too big for the GPU in 16 bits (an 8B reader on an 11 GB card) is loaded in 4 bits"""
    return bool(n_b) and torch.cuda.is_available() and n_b * 2e9 / 2**30 + MARGIN_GB > gpu_free_gb()


def fallback_dtype(n_b: float | None) -> torch.dtype:
    """after float16 overflowed (and that copy was freed): float32 if it fits, else bfloat16 (emulated)"""
    return torch.float32 if not torch.cuda.is_available() or _fits_fp32(n_b) else torch.bfloat16


def has_nan(scores) -> bool:
    """any NaN in generate()'s per-step scores - the sign of a float16 overflow"""
    return any(bool(torch.isnan(s).any()) for s in (scores or ()))


def model_gb(model) -> float:
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 2**30


def park(reader, quantized: bool = False):
    """free the memory a reader's model holds (reader: .model, .device, ._load(), .parked) for the next
    reader: off the GPU into the computer's memory when there is room (back in a second or two), else
    dropped (loaded again from disk on next use); on a CPU-only machine dropped only when memory is short"""
    if getattr(reader, "parked", False) or getattr(reader, "model", None) is None:
        return
    ram = _ram_free_gb()
    if str(reader.device).startswith("cuda"):
        if not quantized and (ram is None or model_gb(reader.model) + 2.0 < ram):
            reader.model.to("cpu")
            reader.parked = "cpu"
        else:
            del reader.model
            reader.model = None
            reader.parked = "dropped"
    elif ram is not None and ram < 8.0:
        del reader.model
        reader.model = None
        reader.parked = "dropped"
    release()


def unpark(reader):
    if reader.parked == "cpu":
        reader.model.to(reader.device)
    elif reader.parked == "dropped":
        reader._load()
    reader.parked = False


def release():
    """give the memory of dropped / moved models back to the GPU"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def describe() -> str:
    if not torch.cuda.is_available():
        return f"CPU ({os.cpu_count()} threads), torch {torch.__version__}"
    p = torch.cuda.get_device_properties(0)
    return (f"{p.name}, {p.total_memory / 2**30:.1f} GB ({gpu_free_gb():.1f} free), compute capability "
            f"{p.major}.{p.minor}, torch {torch.__version__}, CUDA {torch.version.cuda}")
