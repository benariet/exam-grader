"""Renamed: tools/gpu_eval.py (the same parts and more - env, math, hebrew, english, page). This name
still works and runs it with the same arguments."""
import os
import runpy
import sys

sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gpu_eval.py")
runpy.run_path(sys.argv[0], run_name="__main__")
