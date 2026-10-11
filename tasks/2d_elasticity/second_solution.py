import numpy as np


def elastic_energy_density_2d(strains, stiffness_npm, free_components=()):
    """Independent solution by scalar completion of the normal strain square."""
    if any(np.iscomplexobj(value) for value in
           (strains, stiffness_npm, free_components)):
        raise ValueError("Inputs must be real")
    try:
        strain = np.asarray(strains, dtype=float)
        coefficients = np.asarray(stiffness_npm, dtype=float)
        labels = np.asarray(free_components, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Inputs must be numeric") from error
    if (strain.ndim != 2 or strain.shape[1] != 3
            or not np.all(np.isfinite(strain))
            or np.any(np.abs(strain) > 0.03)):
        raise ValueError("Invalid imposed strains")
    if coefficients.shape != (4,) or not np.all(np.isfinite(coefficients)):
        raise ValueError("Invalid stiffness coefficients")
    if (labels.ndim != 1 or not np.all(np.isfinite(labels))
            or np.any(labels != np.floor(labels))
            or np.any((labels < 0) | (labels > 2))
            or np.unique(labels).size != labels.size):
        raise ValueError("Invalid free component indices")

    free = np.zeros(3, dtype=bool)
    free[labels.astype(int)] = True
    c11, c22, c12, c66 = coefficients
    if ((free[0] and c11 <= 0) or (free[1] and c22 <= 0)
            or (free[2] and c66 <= 0)):
        raise ValueError("Free curvature must be strictly positive")
    if free[0] and free[1]:
        extended = coefficients.astype(np.longdouble)
        determinant = extended[0]*extended[1] - extended[2]**2
        # Resolve cancellation by the exact principal minor of represented inputs.
        # This branch recomputes an ambiguous sign; it never rejects a small
        # positive curvature by applying a mechanical stability cutoff.
        rounding_bound = 4*np.finfo(np.longdouble).eps*(abs(extended[0]*extended[1])+extended[2]*extended[2])
        if abs(determinant) <= rounding_bound:
            (an, ad), (bn, bd), (kn, kd) = [
                value.as_integer_ratio() for value in coefficients[:3]]
            numerator = an*bn*kd*kd - kn*kn*ad*bd
            denominator = ad*bd*kd*kd
            determinant = np.longdouble(numerator)/np.longdouble(denominator)
        if determinant <= 0:
            raise ValueError("Free normal block must be positive definite")

    relaxed = strain.copy()
    if free[0] and free[1]:
        relaxed[:, :2] = 0.0
    elif free[0]:
        relaxed[:, 0] = -(c12/c11)*relaxed[:, 1]
    elif free[1]:
        relaxed[:, 1] = -(c12/c22)*relaxed[:, 0]
    if free[2]:
        relaxed[:, 2] = 0.0

    x, y, shear = relaxed.T
    stress = np.column_stack((c11*x+c12*y, c12*x+c22*y, c66*shear))
    energy = (0.5*c11*x*x + c12*x*y + 0.5*c22*y*y
              + 0.5*c66*shear*shear) / 16.02176634
    return {"energy_density_ev_ang2": energy,
            "relaxed_strains": relaxed,
            "stress_npm": stress}


def fit_stiffness_2d(strains, delta_energy_ev, area_ang2):
    if any(np.iscomplexobj(value) for value in (strains, delta_energy_ev, area_ang2)):
        raise ValueError("Inputs must be real")
    e = np.asarray(strains, dtype=float)
    energy = np.asarray(delta_energy_ev, dtype=float)
    area = np.asarray(area_ang2, dtype=float)
    if e.ndim != 2 or e.shape[1] != 3 or not np.all(np.isfinite(e)) or np.any(abs(e) > .03):
        raise ValueError('Invalid strains')
    if energy.shape != (len(e),) or not np.all(np.isfinite(energy)):
        raise ValueError('Invalid energies')
    if area.ndim != 0 or not np.isfinite(area) or area <= 0 or len(e) < 4:
        raise ValueError('Invalid area or number of observations')
    # Construct columns as responses of the four constitutive basis matrices.
    basis = np.zeros((4, 3, 3))
    basis[0, 0, 0] = basis[1, 1, 1] = basis[3, 2, 2] = 1.
    basis[2, 0, 1] = basis[2, 1, 0] = 1.
    design = .5 * np.einsum('ni,kij,nj->nk', e, basis, e)
    if np.any(np.linalg.norm(design, axis=0) / np.sqrt(len(e)) < 1e-6 * (1 - 64*np.finfo(float).eps)):
        raise ValueError('Insufficient signal')
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] <= 0 or singular[0] / singular[-1] > 1000 * (1 + 64*np.finfo(float).eps):
        raise ValueError('Unidentifiable design')
    q, r = np.linalg.qr(design, mode='reduced')
    return np.linalg.solve(r, q.T @ (energy * (16.02176634 / float(area))))


def directional_moduli_2d(stiffness_npm, angles_rad):
    """Resolve applied stress in independently derived normal eigenmodes."""
    if any(np.iscomplexobj(value) for value in (stiffness_npm, angles_rad)):
        raise ValueError('Inputs must be real')
    c = np.asarray(stiffness_npm, dtype=float)
    angles = np.asarray(angles_rad, dtype=float)
    if c.shape != (4,) or not np.all(np.isfinite(c)):
        raise ValueError('Invalid stiffness')
    if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError('Invalid directions')
    c11, c22, c12, c66 = c.astype(np.longdouble)
    determinant = c11*c22-c12*c12
    # Resolve cancellation by the exact principal minor of represented inputs.
    # This branch recomputes an ambiguous sign; it never rejects a small
    # positive curvature by applying a mechanical stability cutoff.
    rounding_bound = 4*np.finfo(np.longdouble).eps*(abs(c11*c22)+c12*c12)
    if abs(determinant) <= rounding_bound:
        (an, ad), (bn, bd), (kn, kd) = [
            value.as_integer_ratio() for value in c[:3]]
        numerator = an*bn*kd*kd - kn*kn*ad*bd
        denominator = ad*bd*kd*kd
        determinant = np.longdouble(numerator)/np.longdouble(denominator)
    if min(c11, c22, c66) <= 0 or determinant <= 0:
        raise ValueError('Unstable material')

    # The large eigenvalue is evaluated without cancellation; the determinant
    # then gives the soft eigenvalue without subtracting nearly equal numbers.
    separation = np.hypot(c11-c22, 2*c12)
    hard = (c11+c22+separation)/2
    soft = determinant/hard
    # Choose a well-scaled column of the hard-mode projector. Algebraic
    # normalization preserves tiny eigenvector components near an axis.
    if c12 == 0:
        cosine, sine = ((np.longdouble(1), np.longdouble(0)) if c11 >= c22
                        else (np.longdouble(0), np.longdouble(1)))
    else:
        vector_x, vector_y = ((hard-c22, c12) if c11 >= c22
                              else (c12, hard-c11))
        length = np.hypot(vector_x, vector_y)
        cosine, sine = vector_x/length, vector_y/length
    co = np.cos(angles.astype(np.longdouble))
    si = np.sin(angles.astype(np.longdouble))
    normal_x, normal_y = co*co, si*si
    applied_hard = normal_x*cosine+normal_y*sine
    applied_soft = -normal_x*sine+normal_y*cosine
    transverse_hard = normal_y*cosine+normal_x*sine
    transverse_soft = -normal_y*sine+normal_x*cosine
    shear_work = normal_x*normal_y/c66
    axial = applied_hard**2/hard + applied_soft**2/soft + shear_work
    transverse = (applied_hard*transverse_hard/hard
                  + applied_soft*transverse_soft/soft - shear_work)
    return np.asarray(1/axial, dtype=float), np.asarray(-transverse/axial, dtype=float)


def analyze_elastic_2d(strains, delta_energy_ev, area_ang2, angles_rad):
    if np.iscomplexobj(angles_rad):
        raise ValueError("Inputs must be real")
    angles = np.asarray(angles_rad, dtype=float)
    if angles.ndim != 1 or angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError('Invalid angular sampling')
    c = fit_stiffness_2d(strains, delta_energy_ev, area_ang2)
    ld = c.astype(np.longdouble)
    determinant = ld[0]*ld[1]-ld[2]*ld[2]
    # Resolve cancellation by the exact principal minor of represented inputs.
    # This branch recomputes an ambiguous sign; it never rejects a small
    # positive curvature by applying a mechanical stability cutoff.
    rounding_bound = 4*np.finfo(np.longdouble).eps*(abs(ld[0]*ld[1])+ld[2]*ld[2])
    if abs(determinant) <= rounding_bound:
        (an, ad), (bn, bd), (kn, kd) = [
            value.as_integer_ratio() for value in c[:3]]
        numerator = an*bn*kd*kd - kn*kn*ad*bd
        denominator = ad*bd*kd*kd
        determinant = np.longdouble(numerator)/np.longdouble(denominator)
    stable = bool(min(ld[0], ld[1], ld[3]) > 0 and determinant > 0)
    if not stable:
        young = np.full(angles.shape, np.nan)
        poisson = young.copy()
        emin = emax = ratio = amin = amax = float('nan')
    else:
        young, poisson = directional_moduli_2d(c, angles)
        a, b, k, shear = c.astype(np.longdouble)
        hard = (a+b+np.hypot(a-b,2*k))/2
        soft = determinant/hard
        # An algebraic normalized eigenvector avoids angle roundoff when the
        # hard normal mode lies extremely close to a principal axis.
        if k == 0:
            cosine, sine = ((np.longdouble(1), np.longdouble(0)) if a >= b
                            else (np.longdouble(0), np.longdouble(1)))
        else:
            vx, vy = ((hard-b, k) if a >= b else (k, hard-a))
            length = np.hypot(vx, vy)
            cosine, sine = vx/length, vy/length

        def reduced_slope(theta):
            co, si = np.cos(np.longdouble(theta)), np.sin(np.longdouble(theta))
            u, v = co*co, si*si
            applied_hard = cosine*u+sine*v
            applied_soft = -sine*u+cosine*v
            xx = cosine*applied_hard/hard-sine*applied_soft/soft
            yy = sine*applied_hard/hard+cosine*applied_soft/soft
            # Derivative of longitudinal strain, with sin(2theta) removed.
            return 2*(yy-xx)+(u-v)/shear

        candidates = [0., float(np.pi/2)]
        left, right = 0., float(np.pi/2)
        fl, fr = reduced_slope(left), reduced_slope(right)
        if fl*fr < 0:
            for _ in range(64):
                middle = (left+right)/2
                fm = reduced_slope(middle)
                if fm == 0:
                    left = right = middle
                    break
                if (fl < 0) == (fm < 0):
                    left, fl = middle, fm
                else:
                    right = middle
            candidates.append((left+right)/2)
        candidates = np.sort(candidates)
        # Determine discrete ties at the fitted constitutive model, without
        # amplifying permissible sampled-helper errors into an angle decision.
        co = np.cos(candidates.astype(np.longdouble))
        si = np.sin(candidates.astype(np.longdouble))
        u, v = co*co, si*si
        applied_hard = cosine*u+sine*v
        applied_soft = -sine*u+cosine*v
        values = 1/(applied_hard**2/hard+applied_soft**2/soft+u*v/shear)
        emin, emax = float(min(values)), float(max(values))
        amin = float(candidates[np.flatnonzero(abs(values-emin) <= 1e-10*abs(emin))[0]])
        amax = float(candidates[np.flatnonzero(abs(values-emax) <= 1e-10*abs(emax))[0]])
        ratio = float(emax/emin)
    return {'stiffness_npm': c, 'born_stable': stable,
            'young_npm': young, 'poisson': poisson,
            'young_min_npm': emin, 'young_max_npm': emax,
            'anisotropy_ratio': ratio,
            'theta_young_min_rad': amin, 'theta_young_max_rad': amax}