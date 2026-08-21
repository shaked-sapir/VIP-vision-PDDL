# VENDOR MODIFICATION (see ../UPSTREAM.md): upstream's third line is
# `from . import main`, which pulls in the argparse CLI layer
# (`dl/main/common.py`, `dl/main/rosame_full.py`). That layer is not vendored —
# our adapter replaces it — so importing it here would fail.
from . import util
from . import model
