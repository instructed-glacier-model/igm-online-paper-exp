"""Save native IGM states and metrics; do not solve or evolve the model."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "_shared"))
from igm_online.native import active


def initialize(cfg, state):
    observer = active()
    try:
        observer.initialized(cfg, state)
    except BaseException as error:
        observer.fail(error)
        raise


def update(cfg, state):
    observer = active()
    try:
        observer.observe(cfg, state)
    except BaseException as error:
        observer.fail(error)
        raise


def finalize(cfg, state):
    observer = active()
    try:
        observer.finish(state)
    except BaseException as error:
        observer.fail(error)
        raise
