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
        normal_determinant = extended[0]*extended[1]-extended[2]**2
        # Resolve cancellation by the exact principal minor of represented inputs.
        # This branch recomputes an ambiguous sign; it never rejects a small
        # positive curvature by applying a mechanical stability cutoff.
        rounding_bound = 4*np.finfo(np.longdouble).eps*(abs(extended[0]*extended[1])+extended[2]*extended[2])
        if abs(normal_determinant) <= rounding_bound:
            (an, ad), (bn, bd), (kn, kd) = [
                value.as_integer_ratio() for value in coefficients[:3]]
            numerator = an*bn*kd*kd - kn*kn*ad*bd
            denominator = ad*bd*kd*kd
            normal_determinant = np.longdouble(numerator)/np.longdouble(denominator)
        if (np.any(np.diag(free_block) <= 0)
                or (0 in free and 1 in free
                    and normal_determinant <= 0)):
            raise ValueError("Free stiffness block must be positive definite")
        try:
            coupling = stiffness[np.ix_(free, prescribed)]
            drive = -(relaxed[:, prescribed] @ coupling.T)
            # With zero forcing the unique equilibrium is exactly zero; avoid
            # a rounded factorization of a nearly singular, admissible block.
            if np.any(drive):
                relaxed[:, free] = np.linalg.solve(free_block, drive.T).T
            else:
                relaxed[:, free] = 0.0
        except np.linalg.LinAlgError as error:
            raise ValueError("Free stiffness block must be positive definite") from error

    stress = relaxed @ stiffness.T
    energy = 0.5 * np.sum(relaxed * stress, axis=1) / 16.02176634
    return {"energy_density_ev_ang2": energy,
            "relaxed_strains": relaxed,
            "stress_npm": stress}
