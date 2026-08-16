"""Source-agnostic trace generation: walk or read a trace, cut it, emit a corpus.

Three stages, composed by ``benchmark/data_generator.py``:

    source  → List[TraceStep]     (sources.py)
    cutter  → List[Window]        (cutter.py, pure)
    emitter → problemN/ folders   (emitter.py)

``TraceStep`` is the only type the three stages share, so neither the cutter nor
the emitter branches on where the trace came from.
"""

from src.trace_generation.trace_step import TraceStep, TraceState

__all__ = ["TraceStep", "TraceState"]
