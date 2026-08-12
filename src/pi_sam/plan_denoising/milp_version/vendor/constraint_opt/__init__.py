from __future__ import annotations

from .factory import register_problem, resolve

# Register encodings.
#
# VENDOR MODIFICATION (see ../UPSTREAM.md): the Gurobi encoder is registered
# only when gurobipy is importable, so the CP-SAT path works without a Gurobi
# license. Upstream imports both unconditionally. Installing gurobipy makes
# "mip-gurobi" available again with no code changes.
from .cp_sat import build_cp_sat_encoding

register_problem("cp-sat", build_cp_sat_encoding)

try:
    from .mip_gurobi import build_mip_gurobi_encoding
except ImportError:  # gurobipy not installed — CP-SAT remains available
    build_mip_gurobi_encoding = None
else:
    register_problem("mip-gurobi", build_mip_gurobi_encoding)
