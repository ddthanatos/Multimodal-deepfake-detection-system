import os
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
import runpy
import pipeline
import clip_patch
runpy.run_path("app.py", run_name="__main__")