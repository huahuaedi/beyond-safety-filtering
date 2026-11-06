#!/usr/bin/env python3
"""Run an official inference script unchanged after the class-level fix that the released anchor code needs.

The released Emu3QFormer.__init__ assigns self.anchor before self._anchor_setup_done exists, so the property setter raises
whenever the anchor head is enabled. A class-level default only records the flag; __init__ later builds the anchor modules.
Usage (under torchrun): run_inference_anchor_patch.py <script.py> [script args...]
"""
import os
import runpy
import sys

repo = os.environ["DRIVEVLA_REPO_ROOT"]
sys.path.insert(0, os.path.join(repo, "inference", "vla"))
from config import setup_paths_early  # noqa: E402

setup_paths_early()
import emu3.mllm.modeling_emu3 as modeling  # noqa: E402
from emu3.mllm import Emu3QFormer  # noqa: E402

print("modeling_emu3 from:", modeling.__file__, flush=True)
Emu3QFormer._anchor_setup_done = True
script = sys.argv[1]
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
