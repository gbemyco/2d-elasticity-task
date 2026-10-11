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
    # Resolve cancellation by the exact principal minor of represented inputs.
    # This branch recomputes an ambiguous sign; it never rejects a small
    # positive curvature by applying a mechanical stability cutoff.
    rounding_bound = 4*np.finfo(np.longdouble).eps*(abs(c11*c22)+c12*c12)
    if abs(det) <= rounding_bound:
        (an, ad), (bn, bd), (kn, kd) = [
            value.as_integer_ratio() for value in c[:3]]
        numerator = an*bn*kd*kd - kn*kn*ad*bd
        denominator = ad*bd*kd*kd
        det = np.longdouble(numerator)/np.longdouble(denominator)
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
