import numpy as np
from .full_graph import PAPER_TARGETS
from .label_fraction import target_distance

def above_paper_checks(dataset, report, minimum_excess=1e-05):
    if minimum_excess <= 0:
        raise ValueError('Use a positive margin to require strictly greater metrics')
    return {key: bool(np.isfinite(report['test'][key]) and report['test'][key] >= target + minimum_excess) for key, target in PAPER_TARGETS[dataset].items()}

def constrained_rank(dataset, report, minimum_excess=1e-05):
    feasible = all(above_paper_checks(dataset, report, minimum_excess).values())
    distance = target_distance(dataset, report)
    return (0 if feasible else 1, distance if np.isfinite(distance) else float('inf'))
