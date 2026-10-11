import numpy as np


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
