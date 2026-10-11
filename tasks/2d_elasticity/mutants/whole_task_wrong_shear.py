"""Scientific mutant: doubles the engineering-shear energy coefficient."""
import numpy as np


def elastic_energy_density_2d(strains, stiffness_npm, free_components=()):
    """Minimize harmonic energy over freely relaxing engineering strains.

    Prescribed strain components are held fixed. A free component carries
    zero conjugate stress, so its equilibrium follows from a restricted
    stiffness solve. Only the free block must be positive definite: held
    directions may retain negative curvature for stability diagnosis.
    """
    if any(np.iscomplexobj(value) for value in
           (strains, stiffness_npm, free_components)):
        raise ValueError("Inputs must be real")
    try:
        strain = np.asarray(strains, dtype=float)
        coefficients = np.asarray(stiffness_npm, dtype=float)
        free_values = np.asarray(free_components, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Inputs must be real numeric arrays") from error

    if (strain.ndim != 2 or strain.shape[1] != 3
            or not np.all(np.isfinite(strain))
            or np.any(np.abs(strain) > 0.03)):
        raise ValueError("Strains must have shape (M, 3) and lie in [-0.03, 0.03]")
    if coefficients.shape != (4,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Stiffness must contain four finite coefficients")
    if (free_values.ndim != 1 or not np.all(np.isfinite(free_values))
            or np.any(free_values != np.floor(free_values))
            or np.any((free_values < 0) | (free_values > 2))
            or np.unique(free_values).size != free_values.size):
        raise ValueError("Free components must be unique indices 0, 1, or 2")

    c11, c22, c12, c66 = coefficients
    c66 *= 2
    stiffness = np.array([[c11, c12, 0.0],
                          [c12, c22, 0.0],
                          [0.0, 0.0, c66]])
    free = free_values.astype(int)
    relaxed = strain.copy()
    if free.size:
        prescribed = np.setdiff1d(np.arange(3), free)
        free_block = stiffness[np.ix_(free, free)]
        # Mechanical admissibility is a strict principal-minor decision.
        # A rounded Cholesky pivot alone can misclassify a singular normal block.
        extended = coefficients.astype(np.longdouble)
        if (np.any(np.diag(free_block) <= 0)
                or (0 in free and 1 in free
                    and extended[0]*extended[1] - extended[2]**2 <= 0)):
            raise ValueError("Free stiffness block must be positive definite")
        try:
            coupling = stiffness[np.ix_(free, prescribed)]
            drive = -(relaxed[:, prescribed] @ coupling.T)
            relaxed[:, free] = np.linalg.solve(free_block, drive.T).T
        except np.linalg.LinAlgError as error:
            raise ValueError("Free stiffness block must be positive definite") from error

    stress = relaxed @ stiffness.T
    energy = 0.5 * np.sum(relaxed * stress, axis=1) / 16.02176634
    return {"energy_density_ev_ang2": energy,
            "relaxed_strains": relaxed,
            "stress_npm": stress}


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
                              x*y_normal, shear*shear))
    rounding = 64 * np.finfo(float).eps
    if np.any(np.sqrt(np.mean(design*design, axis=0)) < 1e-6 * (1 - rounding)):
        raise ValueError("Insufficient quadratic strain signal")
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= 0 or singular[0] / singular[-1] > 1000 * (1 + rounding):
        raise ValueError("Unidentifiable or ill-conditioned design")
    return np.linalg.lstsq(design, y * (16.02176634 / float(a)), rcond=None)[0]


def directional_moduli_2d(stiffness_npm, angles_rad):
    """Engineering compliance projected onto longitudinal and transverse axes."""
    if any(np.iscomplexobj(value) for value in (stiffness_npm, angles_rad)):
        raise ValueError("Inputs must be real")
    c = np.asarray(stiffness_npm, dtype=float)
    theta = np.asarray(angles_rad, dtype=float)
    if c.shape != (4,) or not np.all(np.isfinite(c)):
        raise ValueError("Invalid stiffness")
    if theta.ndim != 1 or theta.size == 0 or not np.all(np.isfinite(theta)):
        raise ValueError("Invalid angles")
    # Extended arithmetic protects the subtraction in the normal-block determinant.
    c11, c22, c12, c66 = c.astype(np.longdouble)
    det = c11*c22 - c12*c12
    if min(c11, c22, c66) <= 0 or det <= 0:
        raise ValueError("Born-unstable stiffness")
    # Evaluate trigonometry before rounding to binary64: a nearly soft normal
    # mode can amplify a small angular error in transverse strain.
    theta_extended = theta.astype(np.longdouble)
    u = np.cos(theta_extended)**2
    v = np.sin(theta_extended)**2
    coupling = c12/c11
    normal_pivot = det/c11
    residual = v-coupling*u
    # Completing the normal-compliance square avoids subtracting large positive
    # terms when loading nearly orthogonal to a soft normal eigenmode.
    axial = u*u/c11 + residual*residual/normal_pivot + u*v/c66
    transverse = u*v/c11 + residual*(u-coupling*v)/normal_pivot - u*v/c66
    return np.asarray(1/axial, dtype=float), np.asarray(-transverse/axial, dtype=float)


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
        s11, s22, s12, s66 = c22/det, c11/det, -c12/det, 1/c66
        # In t=cos(theta)^2 the longitudinal compliance is a quadratic.
        aa = s11+s22-2*s12-s66
        bb = 2*s12+s66-2*s22
        candidates = [0.0, float(np.pi/2)]
        if aa != 0:
            t = -bb/(2*aa)
            if 0 < t < 1:
                candidates.append(float(np.arccos(np.sqrt(t))))
        candidates = np.sort(candidates)
        # Discrete tie selection must use the fitted constitutive model itself:
        # the allowed error in a sampled-response helper exceeds the tie scale.
        critical = candidates.astype(np.longdouble)
        u, v = np.cos(critical)**2, np.sin(critical)**2
        # Completing the square avoids subtraction of large normal-compliance
        # terms in the stiff direction of an almost-soft normal block.
        residual = v - (c12/c11)*u
        values = 1 / (u*u/c11 + residual*residual/(det/c11) + u*v/c66)
        ymin, ymax = float(np.min(values)), float(np.max(values))
        # Public tie rule: relative 1e-10, including isotropic roundoff.
        lo = np.flatnonzero(np.abs(values-ymin) <= 1e-10*abs(ymin))[0]
        hi = np.flatnonzero(np.abs(values-ymax) <= 1e-10*abs(ymax))[0]
        tmin, tmax = float(candidates[lo]), float(candidates[hi])
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