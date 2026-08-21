# VENDOR MODIFICATION (see ../../UPSTREAM.md): upstream re-exports the argparse
# CLI (`from . import common`, `from . import rosame_full`, `from .common import
# main`). Neither module is vendored — our adapter replaces the `.npy` loader
# and we do not want a hyperparameter sweep running inside a fold. The package
# survives only to host `normalization.py`.
