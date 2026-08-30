"""Look-by-look monitoring of a running experiment.

Replays an observation log the way an analyst reads a dashboard -- one
checkpoint at a time -- and answers the question the rest of this package
cannot: given everything seen so far, may we stop, and what would a naive
peeker have done differently?
"""

from abtestlab.monitor.guards import DEFAULT_SRM_THRESHOLD, SrmVerdict, check_srm
from abtestlab.monitor.looks import Look, LookSequenceError, build_looks
from abtestlab.monitor.policy import StopDecision, decide_always_valid, decide_naive_fixed
from abtestlab.monitor.run import (
    KEEP_RUNNING,
    REFUSED_SRM,
    ROLL_BACK,
    SHIP,
    MonitorReport,
    run_monitor,
)
from abtestlab.monitor.sequential import LookEvaluation, evaluate_look, evaluate_looks

__all__ = [
    "DEFAULT_SRM_THRESHOLD",
    "KEEP_RUNNING",
    "REFUSED_SRM",
    "ROLL_BACK",
    "SHIP",
    "Look",
    "LookEvaluation",
    "LookSequenceError",
    "MonitorReport",
    "SrmVerdict",
    "StopDecision",
    "build_looks",
    "check_srm",
    "decide_always_valid",
    "decide_naive_fixed",
    "evaluate_look",
    "evaluate_looks",
    "run_monitor",
]
