# Scientific error: estimates global extrema from the supplied sample grid.
import numpy as np

def analyze_elastic_2d(strains, delta_energy_ev, area_ang2, angles_rad):
    """Infer stiffness and find global elastic extrema over all loading directions."""
    if any(np.iscomplexobj(value) for value in (angles_rad,)):
        raise ValueError("Inputs must be real")
    theta = np.asarray(angles_rad, dtype=float)
    if theta.ndim != 1 or theta.size == 0 or not np.all(np.isfinite(theta)):
        raise ValueError("Invalid angles")
    c = fit_stiffness_2d(strains, delta_energy_ev, area_ang2)
    c11, c22, c12, c66 = c.astype(np.longdouble)
    det = c11*c22-c12*c12
    stable = bool(min(c11, c22, c66) > 0 and det > 0)
    if stable:
        young, poisson = directional_moduli_2d(c, theta)
        # Wrong approximation: samples replace the continuous extrema.
        ymin, ymax = float(np.min(young)), float(np.max(young))
        tmin = float(theta[np.argmin(young)])
        tmax = float(theta[np.argmax(young)])
        ratio = float(ymax/ymin)
    else:
        young = np.full(theta.shape, np.nan)
        poisson = young.copy()
        ymin = ymax = ratio = tmin = tmax = float('nan')
    return {'stiffness_npm': c, 'born_stable': stable,
            'young_npm': young, 'poisson': poisson,
            'young_min_npm': ymin, 'young_max_npm': ymax,
            'anisotropy_ratio': ratio,
            'theta_young_min_rad': tmin, 'theta_young_max_rad': tmax}