
# --- test case 0 ---
import numpy as np

# Isotropic elasticity: recover analytical constants
E, nu = 200.0, 0.25
C = np.array([E/(1-nu**2), E/(1-nu**2),
              nu*E/(1-nu**2), E/(2*(1+nu))])

e = np.array([
    [0.01,0,0], [-0.01,0,0], [0,0.012,0], [0,-0.012,0],
    [0.01,0.01,0], [0.01,-0.01,0], [0,0,0.018], [0,0,-0.018]
])

x, y, g = e.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
area = 150.0

result = fit_stiffness_2d(e, area*(X @ C)/16.02176634, area)
assert np.allclose(result, C, rtol=1e-8, atol=1e-5)


# --- test case 1 ---
import numpy as np

# Anisotropic elasticity with deterministic energy noise
C = np.array([160., 90., 25., 45.])
e = np.array([
    [0.01,0,0], [-0.01,0,0], [0,0.015,0], [0,-0.015,0],
    [0.01,0.01,0], [-0.01,-0.01,0], [0.01,-0.01,0],
    [0,0,0.02], [0,0,-0.02], [0.008,-0.012,0.01]
])

x, y, g = e.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
area = 166.4
noise = 1e-6*np.array([1,-1,0.5,-0.5,1.5,-1.5,0.75,-0.75,1.25,-0.25])
energies = area*(X @ C)/16.02176634 + noise

Q, R = np.linalg.qr(X, mode="reduced")
expected = np.linalg.solve(R, Q.T @ (energies*16.02176634/area))
assert np.allclose(fit_stiffness_2d(e, energies, area),
                   expected, rtol=1e-5, atol=1e-4)


# --- test case 2 ---
import numpy as np

# Rank-deficient strain design
e = np.array([[0.01,0,0], [-0.01,0,0],
              [0.02,0,0], [-0.02,0,0]])

try:
    fit_stiffness_2d(e, [0.1,0.1,0.4,0.4], 100.0)
except ValueError:
    pass
else:
    raise AssertionError("Rank-deficient design must fail")


# --- test case 3 ---
import numpy as np

# Mixed strains, negative coupling and -3% boundary
C = np.array([180., 70., -18., 52.])
e = np.array([
    [0,0,0], [0.006,0,0], [-0.019,0,0], [0,0.008,0],
    [0,-0.027,0], [0.015,0.01,0], [0.017,-0.015,0],
    [-0.013,-0.011,0.012], [0,0,-0.03], [0,0,0.009],
    [0.02,0.006,-0.014], [-0.011,0.023,0.018]
])

x, y, g = e.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
result = fit_stiffness_2d(e, 197*(X @ C)/16.02176634, 197.0)

assert result.shape == (4,)
assert np.allclose(result, C, rtol=1e-8, atol=1e-5)


# --- test case 4 ---
import numpy as np

# Unresolvable shear signal
e = np.array([
    [0.01,0,0], [-0.01,0,0], [0,0.012,0], [0,-0.012,0],
    [0.01,0.01,0], [0.01,-0.01,0],
    [0,0,1e-7], [0,0,-1e-7]
])

x, y, g = e.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
energy = 150*(X @ np.array([160.,90.,25.,45.]))/16.02176634

try:
    fit_stiffness_2d(e, energy, 150.0)
except ValueError:
    pass
else:
    raise AssertionError("Unresolvable shear strain must fail")


# --- test case 5 ---
import numpy as np

# Nearly collinear strains: ill-conditioned design
e = np.array([
    [0.010,0.01001,0.015], [0.015,0.01503,0.018],
    [-0.012,-0.01198,0.018], [0.025,0.02501,0.016],
    [-0.020,-0.02003,0.010], [0.022,0.02199,0.023],
    [0.018,0.01798,0.011], [-0.027,-0.02702,0.021]
])

x, y, g = e.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
energy = 160*(X @ np.array([140.,80.,20.,55.]))/16.02176634

try:
    fit_stiffness_2d(e, energy, 160.0)
except ValueError:
    pass
else:
    raise AssertionError("Ill-conditioned strain design must fail")


# --- test case 6 ---
import numpy as np

# Supercell scaling: stiffness is independent of cell area
C = np.array([280., 110., 40., 76.])
e = np.array([
    [0.008,0,0], [-0.013,0,0], [0,0.01,0], [0,-0.019,0],
    [0.012,0.012,0], [0.012,-0.012,0],
    [0,0,0.015], [0,0,-0.021], [0.014,-0.006,0.016]
])

x, y, g = e.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
U = (X @ C)/16.02176634

c1 = fit_stiffness_2d(e, 125*U, 125.0)
c2 = fit_stiffness_2d(e, 500*U, 500.0)

assert np.allclose(c1, C, rtol=1e-8, atol=1e-5)
assert np.allclose(c2, C, rtol=1e-8, atol=1e-5)
assert np.allclose(c1, c2, rtol=2e-8, atol=2e-5)

# --- test case 7 ---
import numpy as np

# Energy observations must correspond one-to-one with strain rows.
e = np.array([[.01,0,0],[-.01,0,0],[0,.012,0],[0,-.012,0],
              [.01,.01,0],[.01,-.01,0],[0,0,.02],[0,0,-.02]])
for y in (np.ones(7), np.ones((8, 1))):
    try:
        fit_stiffness_2d(e, y, 120.)
    except ValueError:
        pass
    else:
        raise AssertionError('Energy array of incorrect shape was accepted')

# --- test case 8 ---
import numpy as np

# Area normalization requires positive physical basal-plane area.
e = np.array([[.01,0,0],[-.01,0,0],[0,.012,0],[0,-.012,0],
              [.01,.01,0],[.01,-.01,0],[0,0,.02],[0,0,-.02]])
for area in (0., -150.):
    try:
        fit_stiffness_2d(e, np.ones(8), area)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid 2D supercell area was accepted')

# --- test case 9 ---
import numpy as np

# A small yet resolvable shear column (RMS around 2e-6) must be accepted.
C = np.array([145., 92., -17., 51.])
e = np.array([[.01,0,0],[-.01,0,0],[0,.012,0],[0,-.012,0],
              [.01,.01,0],[.01,-.01,0],[0,0,.003],[0,0,-.003],
              [.009,-.007,0]])
x,y,g = e.T
X = np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert 1e-6 < np.sqrt(np.mean(X[:,3]**2)) < 3e-6
assert np.linalg.cond(X) < 1000
area = 170.
r = fit_stiffness_2d(e, (area/16.02176634)*(X @ C), area)
assert np.allclose(r, C, rtol=1e-8, atol=1e-5)

# --- test case 10 ---
import numpy as np

# A moderate, accepted condition number must not be rejected as unstable fitting.
C = np.array([210., 88., 31., 65.])
e = np.array([
    [.019452,.019443,.006312],[.008238,.009893,.015446],
    [.016212,.015763,.014509],[-.006961,-.007394,-.006217],
    [-.017226,-.016576,.013561],[.010738,.009069,-.001338],
    [-.009086,-.007733,-.004333],[-.002488,-.003562,.007128],
    [.006265,.007258,.004691],[-.007809,-.009500,-.008979],
    [.003154,.001557,-.000494],[-.009833,-.008106,.003130]])
x,y,g = e.T
X = np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert 100 < np.linalg.cond(X) < 1000
assert min(np.sqrt(np.mean(X**2,axis=0))) >= 1e-6
r = fit_stiffness_2d(e, 180*(X @ C)/16.02176634, 180.)
assert np.allclose(r, C, rtol=1e-8, atol=1e-5)

# --- test case 11 ---
import numpy as np
# Four observations are sufficient; shear-column RMS is exactly the allowed endpoint.
a=.002
e=np.array([[a,0,0],[0,a,0],[a,a,0],[0,0,a]])
x,y,g=e.T
X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert np.isclose(np.sqrt(np.mean(X[:,3]**2)),1e-6,rtol=1e-12,atol=1e-18)
C=np.array([170.,95.,-23.,41.]); area=124.
r=fit_stiffness_2d(e,area*(X@C)/16.02176634,area)
assert isinstance(r,np.ndarray) and r.shape==(4,) and r.dtype.kind=='f'
assert np.allclose(r,C,rtol=1e-8,atol=1e-5)

# --- test case 12 ---
import numpy as np
# Just-below-threshold signal must be rejected, not rescued by column normalization.
a=.002
e=np.array([[a,0,0],[0,a,0],[a,a,0],[0,0,a*np.sqrt(.99)]])
x,y,g=e.T; X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert np.linalg.cond(X)<10
assert 0.98e-6 < np.sqrt(np.mean(X[:,3]**2)) < 1e-6
try:
    fit_stiffness_2d(e,125*(X@np.array([180.,90.,20.,45.]))/16.02176634,125.)
except ValueError:
    pass
else:
    raise AssertionError('Below-threshold quadratic signal was accepted')

# --- test case 13 ---
import numpy as np
# Raw design condition number just inside the specified fitting limit.
e=np.array([[0.01, 0.011837425770910136, 0.01], [-0.012, -0.013837425770910136, 0.017], [0.015, 0.013162574229089863, -0.013], [-0.02, -0.018162574229089866, 0.022], [0.023, 0.02116257422908986, 0.008], [-0.009, -0.007162574229089863, -0.019], [0.017, 0.01883742577091014, 0.025], [-0.016, -0.01783742577091014, 0.011]])
x,y,g=e.T; X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert 998.9 < np.linalg.cond(X) < 999.1
assert np.min(np.sqrt(np.mean(X*X,axis=0)))>1e-6
C=np.array([240.,115.,-30.,73.]); energy=135*(X@C)/16.02176634
r=fit_stiffness_2d(e,energy,135.)
assert np.allclose(r,C,rtol=1e-8,atol=1e-5)

# --- test case 14 ---
import numpy as np
# Raw design condition number just outside the specified fitting limit.
e=np.array([[0.01, 0.011835601252181461, 0.01], [-0.012, -0.013835601252181461, 0.017], [0.015, 0.013164398747818539, -0.013], [-0.02, -0.018164398747818538, 0.022], [0.023, 0.021164398747818537, 0.008], [-0.009, -0.007164398747818539, -0.019], [0.017, 0.018835601252181464, 0.025], [-0.016, -0.017835601252181463, 0.011]])
x,y,g=e.T; X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert 1000.9 < np.linalg.cond(X) < 1001.1
assert np.min(np.sqrt(np.mean(X*X,axis=0)))>1e-6
C=np.array([240.,115.,-30.,73.]); energy=135*(X@C)/16.02176634
try:
    fit_stiffness_2d(e,energy,135.)
except ValueError:
    pass
else:
    raise AssertionError('Condition number above 1000 was accepted')

# --- test case 15 ---
import numpy as np
# A nonzero baseline error must be absorbed by the specified no-intercept OLS,
# not silently removed or fit with an intercept. The QR target is independent of SVD.
e=np.array([[.006,0,0],[-.015,0,0],[0,.009,0],[0,-.02,0],
            [.011,.009,0],[.018,-.009,.012],[0,0,.02],[0,0,-.011]])
x,y,g=e.T; X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
C=np.array([170.,90.,21.,36.]); area=173.
energies=area*(X@C)/16.02176634 + .0002
Q,R=np.linalg.qr(X,mode='reduced')
expected=np.linalg.solve(R,Q.T@(energies*16.02176634/area))
actual=fit_stiffness_2d(e,energies,area)
assert np.allclose(actual,expected,rtol=1e-5,atol=1e-4)
assert np.linalg.norm(expected-C)>.1

# --- test case 16 ---
import numpy as np
# A least-squares stiffness diagnostic must retain unstable curvature and negative coupling.
e=np.array([[.012,0,0],[0,.014,0],[.01,.01,0],[.01,-.01,0],
            [0,0,.02],[.008,-.012,.016]])
C=np.array([-130.,-85.,-17.,-22.]); a=132.
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=a*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
assert np.allclose(fit_stiffness_2d(e,energy,a),C,rtol=1e-8,atol=1e-5)

# --- test case 17 ---
import numpy as np
e=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],[.01,-.01,0],
            [0,0,.02],[.008,.009,.01]])
y=np.ones(len(e))
invalid=[]
invalid.extend([(e[:3],y[:3],130.),(e[:, :2],y,130.),(e.ravel(),y,130.),
                (e,y,[130.])])
bad=e.copy(); bad[0,2]=.0301; invalid.append((bad,y,130.))
for strains,energy,area in invalid:
    try:
        fit_stiffness_2d(strains,energy,area)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid fitting input was accepted')

# --- test case 18 ---
import numpy as np
# All columns carry signal, but coincident normal strains cannot identify C11,C22,C12.
e=np.array([[.01,.01,.01],[.02,.02,0],[-.015,-.015,.02],[.025,.025,-.012]])
x,y,g=e.T; X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert np.all(np.sqrt(np.mean(X*X,axis=0))>1e-6)
assert np.linalg.matrix_rank(X)<4
try:
    fit_stiffness_2d(e,140*(X@np.array([150.,90.,20.,35.]))/16.02176634,140.)
except ValueError:
    pass
else:
    raise AssertionError('Rank-deficient but nonzero-column design was accepted')

# --- test case 19 ---
import numpy as np
# Symmetric +/- strains eliminate odd numerical-energy noise in quadratic OLS.
base=np.array([[.01,0,0],[0,.014,0],[.012,.009,0],[.011,-.01,0],[0,0,.019]])
e=np.vstack((base,-base)); C=np.array([210.,110.,25.,55.]); area=141.
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=area*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
energy=energy+e@np.array([.08,-.05,.02])
assert np.allclose(fit_stiffness_2d(e,energy,area),C,rtol=1e-8,atol=1e-5)

# --- test case 20 ---
import numpy as np
# Each observation has equal weight: duplicating a noisy observation changes its leverage.
e=np.array([[.01,0,0],[0,.014,0],[.012,.009,0],[.011,-.01,0],[0,0,.019],
            [.008,.012,-.015],[.015,-.008,.011]])
x,y,g=e.T; X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
C=np.array([145.,93.,-16.,37.]); area=142.
energy=area*(X@C)/16.02176634+1e-4*np.array([1.,-2.,.5,3.,-1.,2.,-.3])
indices=np.array([0,1,2,3,4,5,6,3,3,3])
e2=e[indices]; y2=energy[indices]; X2=X[indices]
Q,R=np.linalg.qr(X2,mode='reduced')
expected=np.linalg.solve(R,Q.T@(y2*16.02176634/area))
actual=fit_stiffness_2d(e2,y2,area)
assert np.allclose(actual,expected,rtol=1e-5,atol=1e-4)


# --- test case 21 ---
import numpy as np

# A symmetric normal-strain block gives an independently derived singular spectrum.
# Its dimensionless Gram eigenvalues are 1/4 and (199 +/- sqrt(34977))/8.
# The isolated shear observation sets the raw spectral condition number to 1000.
a = .03
largest_normal = a*a*np.sqrt((199 + np.sqrt(34977.))/8)
g = np.sqrt(2*largest_normal/1000)
e = np.array([[a,a,0]]*32 + [[a,0,0], [0,a,0], [a,-a,0], [0,0,g]])
x,y,shear = e.T
X = np.column_stack((x*x/2, y*y/2, x*y, shear*shear/2))
expected_singular = np.sort(np.array([
    largest_normal, a*a*np.sqrt((199-np.sqrt(34977.))/8),
    a*a/2, largest_normal/1000]))[::-1]
measured_singular = np.linalg.svd(X, compute_uv=False)
assert np.allclose(measured_singular, expected_singular, rtol=2e-14, atol=1e-18)
# Published relative roundoff guard: 64 times binary64 spacing at 1.
# This bounds an internal arithmetic comparison, not a physical input.
delta = 1.4210854715202004e-14
assert abs(measured_singular[0]/measured_singular[-1]-1000) <= 1000*delta
assert np.min(np.sqrt(np.mean(X*X,axis=0))) > 1e-6
C = np.array([240.,115.,-30.,73.])
area = 135.
r = fit_stiffness_2d(e, area*(X@C)/16.02176634, area)
assert np.allclose(r, C, rtol=1e-8, atol=1e-5)


# --- test case 22 ---
import numpy as np

# A resolvable change beyond roundoff at the conditioning endpoint is inadmissible.
a = .03
largest_normal = a*a*np.sqrt((199 + np.sqrt(34977.))/8)
g = np.sqrt(2*largest_normal/(1000*(1+1e-8)))
e = np.array([[a,a,0]]*32 + [[a,0,0], [0,a,0], [a,-a,0], [0,0,g]])
x,y,shear = e.T
X = np.column_stack((x*x/2,y*y/2,x*y,shear*shear/2))
assert 1000*(1+0.9e-8) < np.linalg.cond(X) < 1000*(1+1.1e-8)
assert np.min(np.sqrt(np.mean(X*X,axis=0))) > 1e-6
try:
    fit_stiffness_2d(e, 135*(X@np.array([240.,115.,-30.,73.]))/16.02176634, 135.)
except ValueError:
    pass
else:
    raise AssertionError('Conditioning outside the roundoff endpoint allowance was accepted')

# --- test case 23 ---
import numpy as np
# Physical strains must be real; discarding an imaginary component changes the data.
e = np.array([[.01,0.,0.], [0.,.012,0.], [.01,.01,0.],
              [.01,-.01,0.], [0.,0.,.02]])
x, y, g = e.T
X = np.column_stack((x*x/2, y*y/2, x*y, g*g/2))
assert np.linalg.matrix_rank(X) == 4 and np.linalg.cond(X) < 1000
assert np.all(np.sqrt(np.mean(X*X,axis=0)) > 1e-6)
energy = 150.*(X@np.array([160.,90.,25.,45.]))/16.02176634
complex_strain = e.astype(complex)
complex_strain[0,0] = .01+1j
try:
    fit_stiffness_2d(complex_strain, energy, 150.)
except ValueError:
    pass
else:
    raise AssertionError('Complex strains were accepted by discarding their imaginary part')

# --- test case 24 ---
import numpy as np

# Isolate the RMS requirement for each normal constitutive coefficient.
for deficient_column in range(3):
    strains=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                      [.01,-.01,0],[0,0,.02]])
    if deficient_column < 2:
        component=strains[:,deficient_column]
        initial_rms=np.sqrt(np.mean((component*component/2)**2))
        strains[:,deficient_column] *= np.sqrt(.99e-6/initial_rms)
    else:
        mixed_amplitude=np.sqrt(.99e-6/np.sqrt(2/5))
        strains[2:4,:2]=[[mixed_amplitude,mixed_amplitude],
                        [mixed_amplitude,-mixed_amplitude]]
    x,y,g=strains.T
    design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
    rms=np.sqrt(np.mean(design*design,axis=0))
    assert np.linalg.matrix_rank(design) == 4
    assert np.linalg.cond(design) < 1000
    assert .98e-6 < rms[deficient_column] < 1e-6
    assert np.all(np.delete(rms,deficient_column) > 1e-6)
    energy=150.*(design@np.array([160.,90.,25.,45.]))/16.02176634
    try:
        fit_stiffness_2d(strains,energy,150.)
    except ValueError:
        pass
    else:
        raise AssertionError('Sub-threshold normal design column was accepted')

# --- test case 25 ---
import numpy as np

# A failed energy calculation must be rejected before regression.
strains=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                  [.01,-.01,0],[0,0,.02]])
x,y,g=strains.T
design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert np.linalg.matrix_rank(design) == 4
assert np.linalg.cond(design) < 1000
assert np.all(np.sqrt(np.mean(design*design,axis=0)) > 1e-6)
valid_energy=150.*(design@np.array([160.,90.,25.,45.]))/16.02176634
for invalid in (np.nan,np.inf,-np.inf):
    energy=valid_energy.copy()
    energy[4]=invalid
    try:
        fit_stiffness_2d(strains,energy,150.)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite energy observation was accepted')

# --- test case 26 ---
import numpy as np

# Constant-magnitude target entries put each normal column at the included
# RMS endpoint: a*a/2=1e-6 for normal strains, A*B=1e-6 for mixed strains.
a=np.sqrt(2e-6)
normal=np.array([[a,0,.01],[a,.01,0],[a,-.01,0],[a,.02,0]])
datasets=(normal,normal[:,[1,0,2]],
          np.array([[.004,.00025,.01],[.00025,.004,0],
                    [.004,-.00025,0],[.00025,-.004,0]]))
stiffness=np.array([160.,90.,25.,45.])
for endpoint_column,strains in enumerate(datasets):
    x,y,g=strains.T
    design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
    rms=np.sqrt(np.mean(design*design,axis=0))
    assert np.linalg.matrix_rank(design) == 4
    assert np.linalg.cond(design) < 1000
    # The comparison budget is the published internal roundoff allowance.
    delta=1.4210854715202004e-14
    assert np.isclose(rms[endpoint_column],1e-6,rtol=delta,atol=0.)
    assert np.all(np.delete(rms,endpoint_column) > 1e-6)
    energy=150.*(design@stiffness)/16.02176634
    fitted=fit_stiffness_2d(strains,energy,150.)
    assert isinstance(fitted,np.ndarray) and fitted.shape == (4,)
    assert np.issubdtype(fitted.dtype,np.floating)
    assert np.allclose(fitted,stiffness,rtol=1e-8,atol=1e-5)

# --- test case 27 ---
import numpy as np
strains=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                  [.01,-.01,0],[0,0,.02]])
x,y,g=strains.T
design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert np.linalg.matrix_rank(design) == 4
assert np.linalg.cond(design) < 1000
assert np.all(np.sqrt(np.mean(design*design,axis=0)) > 1e-6)
area=150.

# Failed or complex physical records cannot be silently replaced or projected.
energy=area*(design@np.array([160.,90.,25.,45.]))/16.02176634
invalid=[]
for bad in (np.nan,np.inf,-np.inf):
    observations=strains.copy()
    observations[0,0]=bad
    invalid.extend([(observations,energy,area),(strains,energy,bad)])
for imaginary in (0.,1.):
    observations=strains.astype(complex)
    observations[0,0]+=1j*imaginary
    energies=energy.astype(complex)
    energies[0]+=1j*imaginary
    invalid.extend([(observations,energy,area),(strains,energies,area),
                    (strains,energy,complex(area,imaginary))])
for observations,energies,basal_area in invalid:
    try:
        fit_stiffness_2d(observations,energies,basal_area)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid physical fitting input was accepted')

# --- test case 28 ---
import numpy as np

# Exercise the published RMS comparison allowance on both sides, independently
# of conditioning. Delta is an internal rounding budget, not a physical datum.
delta=1.4210854715202004e-14
stiffness=np.array([160.,90.,25.,45.])
for multiplier,accepted in ((1-delta/2,True),(1-2*delta,False)):
    a=np.sqrt(2e-6*multiplier)
    normal=np.array([[a,0,.01],[a,.01,0],[a,-.01,0],[a,.02,0]])
    product=1e-6*multiplier
    small=product/.004
    datasets=(normal,normal[:,[1,0,2]],
              np.array([[.004,small,.01],[small,.004,0],
                        [.004,-small,0],[small,-.004,0]]),
              np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                        [0,0,np.sqrt(4e-6*multiplier)]]))
    for column,strains in enumerate(datasets):
        x,y,g=strains.T
        design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
        rms=np.sqrt(np.mean(design*design,axis=0))
        assert np.linalg.matrix_rank(design) == 4
        assert np.linalg.cond(design) < 1000
        assert np.all(np.delete(rms,column) > 1e-6)
        if accepted:
            assert 1e-6*(1-delta) < rms[column] < 1e-6
        else:
            assert rms[column] < 1e-6*(1-delta)
        energy=150.*(design@stiffness)/16.02176634
        if accepted:
            fitted=fit_stiffness_2d(strains,energy,150.)
            assert np.allclose(fitted,stiffness,rtol=1e-8,atol=1e-5)
        else:
            try:
                fit_stiffness_2d(strains,energy,150.)
            except ValueError:
                pass
            else:
                raise AssertionError('Signal below the roundoff allowance was accepted')

# --- test case 29 ---
import numpy as np
strains=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                  [.01,-.01,0],[0,0,.02]])
x,y,g=strains.T
design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert np.linalg.matrix_rank(design) == 4
assert np.linalg.cond(design) < 1000
assert np.all(np.sqrt(np.mean(design*design,axis=0)) > 1e-6)
area=150.

# Inference preserves the coefficient calibration endpoints, including
# unstable curvature; lists are valid real array-like observations.
for stiffness in (np.array([1e4,8000.,1e4,1e4]),
                  np.array([-1e4,1e4,-2500.,-1e4])):
    energy=area*(design@stiffness)/16.02176634
    fitted=fit_stiffness_2d(strains.tolist(),energy.tolist(),area)
    assert isinstance(fitted,np.ndarray) and fitted.shape == (4,)
    assert np.issubdtype(fitted.dtype,np.floating)
    assert np.allclose(fitted,stiffness,rtol=1e-8,atol=1e-5)