"""Hugging Face Space entrypoint.

We keep the original Gradio Blocks definition in app/app.py to avoid a large
refactor (there is already a package directory named 'app'). Because Spaces
expects a top-level app.py, we dynamically load the inner module to avoid the
name clash that would happen with `import app.app` (this file would shadow the
package).

Exports:
  demo: the original Blocks instance (for completeness)
  app:  a queued version (Spaces detects `app` or `demo`)

NOTE: Do NOT call .launch() here; Hugging Face handles serving.
"""
from __future__ import annotations

import importlib.util
import os
import sys

INNER_PATH = os.path.join(os.path.dirname(__file__), "app", "app.py")

if not os.path.isfile(INNER_PATH):  # fail fast with clear message
    raise FileNotFoundError(f"Inner Gradio app file not found at {INNER_PATH}")

spec = importlib.util.spec_from_file_location("_inner_gradio_app", INNER_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not create spec for inner gradio app module")
_inner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = _inner  # register to allow relative imports inside
spec.loader.exec_module(_inner)  # type: ignore

# Expose the Blocks instance
demo = getattr(_inner, "demo")  # The Blocks defined inside app/app.py

# Queue (enable concurrency & progress) and expose as `app` for Spaces
app = demo.queue()
