import numpy as np

def pareto_front_2d_tmax_omin(points: np.ndarray) -> np.ndarray:
    """
    Compute Pareto front for 2D points where:
      - maximize t_norm (x)
      - minimize o_norm (y)

    Returns Pareto points sorted by increasing t_norm.
    """
    if points.size == 0:
        return points

    # sort by descending t (best first)
    pts = points[np.argsort(-points[:, 0])]
    pareto = []
    best_o = np.inf
    for t, o in pts:
        if o < best_o:
            pareto.append((t, o))
            best_o = o
    pareto = np.array(pareto, dtype=float)

    # sort by increasing t for hypervolume integration
    pareto = pareto[np.argsort(pareto[:, 0])]
    return pareto

def hypervolume_2d_tmax_omin(pareto_pts: np.ndarray, ref_t: float, ref_o: float) -> float:
    """
    2D hypervolume for maximize t_norm and minimize o_norm.

    Uses a simple rectangle integration in (t, o) space:
      HV = sum over segments: (t_i - t_{i-1}) * (ref_o - o_i)

    Requirements:
      - ref_t should be <= min(pareto t)
      - ref_o should be >= max(pareto o)
    """
    if pareto_pts.size == 0:
        return 0.0

    hv = 0.0
    prev_t = ref_t
    for t, o in pareto_pts:
        width = t - prev_t
        height = ref_o - o
        if width > 0 and height > 0:
            hv += width * height
        prev_t = t
    return float(hv)