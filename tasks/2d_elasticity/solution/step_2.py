import numpy as np

def fit_stiffness_2d(strains, delta_energy_ev, area_ang2):
    """Ordinary least squares in the physical, unscaled quadratic design."""
    if any(np.iscomplexobj(value) for value in (strains, delta_energy_ev, area_ang2)):
        raise ValueError("Inputs must be real")
    e = np.asarray(strains, dtype=float)
    y = np.asarray(delta_energy_ev, dtype=float)
    a = np.asarray(area_ang2, dtype=float)
    if e.ndim != 2 or e.shape[1] != 3 or not np.all(np.isfinite(e)) or np.any(np.abs(e) > 0.03):
        raise ValueError("Invalid strains")
    if y.shape != (len(e),) or not np.all(np.isfinite(y)):
        raise ValueError("Invalid energies")
    if a.ndim != 0 or not np.isfinite(a) or a <= 0:
        raise ValueError("Invalid area")
    if len(e) < 4:
        raise ValueError("At least four observations are needed")
    # Derive the exact quadratic features directly from engineering strains.
    # Approximate energy outputs would amplify their own tolerance in this fit.
    x, y_normal, shear = e.T
    design = np.column_stack((0.5*x*x, 0.5*y_normal*y_normal,
                              x*y_normal, 0.5*shear*shear))
    rounding = 64 * np.finfo(float).eps
    if np.any(np.sqrt(np.mean(design*design, axis=0)) < 1e-6 * (1 - rounding)):
        raise ValueError("Insufficient quadratic strain signal")
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= 0 or singular[0] / singular[-1] > 1000 * (1 + rounding):
        raise ValueError("Unidentifiable or ill-conditioned design")
    return np.linalg.lstsq(design, y * (16.02176634 / float(a)), rcond=None)[0]