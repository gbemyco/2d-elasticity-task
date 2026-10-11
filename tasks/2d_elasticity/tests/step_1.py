# --- test case 0 ---
import numpy as np
# Fully prescribed uniaxial and biaxial work.
c = np.array([160., 90., 25., 45.])
e = np.array([[.01, 0., 0.], [0., -.02, 0.], [.01, -.01, 0.]])
r = elastic_energy_density_2d(e, c)
expected = np.array([.5*160*.01**2, .5*90*.02**2,
                     .5*(160+90)*.01**2-25*.01**2])/16.02176634
assert type(r) is dict and set(r) == {'energy_density_ev_ang2', 'relaxed_strains', 'stress_npm'}
for key, shape in [('energy_density_ev_ang2', (3,)), ('relaxed_strains', (3,3)), ('stress_npm', (3,3))]:
    assert isinstance(r[key], np.ndarray) and r[key].dtype.kind == 'f' and r[key].shape == shape
assert np.allclose(r['energy_density_ev_ang2'], expected, rtol=1e-10, atol=1e-12)
assert np.allclose(r['relaxed_strains'], e, rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], [[1.6,.25,0.],[-.5,-1.8,0.],[1.35,-.65,0.]], rtol=1e-9, atol=1e-8)

# --- test case 1 ---
import numpy as np
# Engineering shear is work conjugate to tau_xy, not twice tau_xy.
c = [160., 90., 25., 45.]
e = [[0.,0.,.02],[0.,0.,-.02]]
r = elastic_energy_density_2d(e, c)
assert np.allclose(r['energy_density_ev_ang2'], .5*45*.02**2/16.02176634, rtol=1e-10, atol=1e-12)
assert np.allclose(r['stress_npm'], [[0.,0.,.9],[0.,0.,-.9]], rtol=1e-9, atol=1e-8)

# --- test case 2 ---
import numpy as np
# Included small-strain endpoints and malformed strain rejection.
c = [100., 100., 20., 40.]
r = elastic_energy_density_2d([[.03,0.,0.],[-.03,0.,0.]], c)
assert np.allclose(r['energy_density_ev_ang2'], .5*100*.03**2/16.02176634, rtol=1e-10, atol=1e-12)
assert np.allclose(r['stress_npm'], [[3.,.6,0.],[-3.,-.6,0.]], rtol=1e-9, atol=1e-8)
for invalid in ([[.031,0.,0.]], [0.01,0.,0.], np.zeros((2,2)), np.zeros((1,3,1))):
    try:
        elastic_energy_density_2d(invalid, c)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid strains must raise ValueError')

# --- test case 3 ---
import numpy as np
# Zero energy and parity under simultaneous strain reversal.
c = [142., 86., -19., 33.]
e = np.array([[0.,0.,0.],[.012,-.009,.011],[-.012,.009,-.011]])
r = elastic_energy_density_2d(e, c)
expected = (.5*142*.012**2+.5*86*.009**2+(-19)*.012*(-.009)+.5*33*.011**2)/16.02176634
assert np.isclose(r['energy_density_ev_ang2'][0], 0., rtol=1e-10, atol=1e-12)
assert np.allclose(r['energy_density_ev_ang2'][1:], expected, rtol=1e-10, atol=1e-12)
assert np.allclose(r['stress_npm'][1], -r['stress_npm'][2], rtol=2e-9, atol=2e-8)

# --- test case 4 ---
import numpy as np
# Four one-dimensional stiffness coefficients are required.
for c in ([120.,70.,20.], [[120.,70.],[20.,35.]], np.zeros((4,1))):
    try:
        elastic_energy_density_2d([[.01,0.,0.]], c)
    except ValueError:
        pass
    else:
        raise AssertionError('Malformed stiffness accepted')

# --- test case 5 ---
import numpy as np
# Empty batches and prescribed unstable curvature retain their meaning.
r = elastic_energy_density_2d(np.empty((0,3)), [20.,30.,2.,-4.])
assert r['energy_density_ev_ang2'].shape == (0,)
assert r['relaxed_strains'].shape == r['stress_npm'].shape == (0,3)
e = np.array([[0.,0.,.03],[.03,-.03,-.03],[-.03,.03,.03]])
r = elastic_energy_density_2d(e, [-12.,-8.,3.,-4.])
assert np.allclose(r['energy_density_ev_ang2'], np.array([-.0018,-.0135,-.0135])/16.02176634, rtol=1e-10, atol=1e-12)
assert np.allclose(r['relaxed_strains'], e, rtol=1e-9, atol=1e-10)

# --- test case 6 ---
import numpy as np
# Harmonic work and homogeneous scaling with independent stress construction.
e = np.array([[.004,-.006,.008],[-.005,.003,-.007]])
c = np.array([260.,130.,-37.,62.])
stress = np.column_stack((c[0]*e[:,0]+c[2]*e[:,1], c[2]*e[:,0]+c[1]*e[:,1], c[3]*e[:,2]))
r = elastic_energy_density_2d(e, c)
r2 = elastic_energy_density_2d(2*e, c)
assert np.allclose(r['stress_npm'], stress, rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], .5*np.sum(stress*e,axis=1)/16.02176634, rtol=1e-10, atol=1e-12)
assert np.allclose(r2['energy_density_ev_ang2'], 4*r['energy_density_ev_ang2'], rtol=2e-10, atol=5e-12)

# --- test case 7 ---
import numpy as np
# Free transverse contraction recovers Young modulus and Poisson ratio.
c = [200.,100.,40.,30.]
e = np.array([[.01,.02,-.025],[-.01,-.03,.017]])
r = elastic_energy_density_2d(e, c, [1,2])
assert np.allclose(r['relaxed_strains'], [[.01,-.004,0.],[-.01,.004,0.]], rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], [[1.84,0.,0.],[-1.84,0.,0.]], rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], .0092/16.02176634, rtol=1e-10, atol=1e-12)

# --- test case 8 ---
import numpy as np
# Auxetic free expansion and asymmetric relaxation along the other axis.
c = [120.,80.,-20.,35.]
r = elastic_energy_density_2d([[.008,-.03,.01]], c, [1])
assert np.allclose(r['relaxed_strains'], [[.008,.002,.01]], rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], [[.92,0.,.35]], rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], (.5*115*.008**2+.5*35*.01**2)/16.02176634, rtol=1e-10, atol=1e-12)
r = elastic_energy_density_2d([[-.03,.012,.005]], c, [0])
assert np.allclose(r['relaxed_strains'], [[.002,.012,.005]], rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], [[0.,.92,.175]], rtol=1e-9, atol=1e-8)

# --- test case 9 ---
import numpy as np
# Index order is irrelevant; free guesses never prescribe loads.
c = [170.,90.,22.,47.]
a = elastic_energy_density_2d([[.006,.03,-.02]], c, [2,1])
b = elastic_energy_density_2d([[.006,-.029,.028]], c, [1.,2.])
for key, atol, rtol in [('energy_density_ev_ang2',2e-12,2e-10),('relaxed_strains',2e-10,2e-9),('stress_npm',2e-8,2e-9)]:
    assert np.allclose(a[key], b[key], atol=atol, rtol=rtol)
assert np.allclose(a['relaxed_strains'], [[.006,-22*.006/90,0.]], rtol=1e-9, atol=1e-10)

# --- test case 10 ---
import numpy as np
# An unloaded fully free stable sheet has zero strain, stress, and stored energy.
c = [140.,100.,25.,35.]
r = elastic_energy_density_2d([[.02,-.01,.03],[-.015,.012,-.018]], c, [2,0,1])
assert np.allclose(r['relaxed_strains'], 0., rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], 0., rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], 0., rtol=1e-10, atol=1e-12)
r = elastic_energy_density_2d(np.empty((0,3)), c, [2,0,1])
assert r['energy_density_ev_ang2'].shape == (0,) and r['relaxed_strains'].shape == r['stress_npm'].shape == (0,3)

# --- test case 11 ---
import numpy as np
# A constrained minimum may exist while a held direction has negative curvature.
c = [-100.,80.,20.,-2.]
r = elastic_energy_density_2d([[.01,.03,.012]], c, [1])
assert np.allclose(r['relaxed_strains'], [[.01,-.0025,.012]], rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], [[-1.05,0.,-.024]], rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], (-.5*105*.01**2-.012**2)/16.02176634, rtol=1e-10, atol=1e-12)
assert r['energy_density_ev_ang2'][0] < 0.

# --- test case 12 ---
import numpy as np
# A nonpositive free curvature has no unique bounded minimum.
for c, free in [([100.,80.,20.,0.],[2]), ([100.,80.,20.,-1.],[2]),
                ([100.,80.,90.,30.],[0,1]), ([-100.,80.,20.,30.],[0]),
                ([100.,100.,100.,30.],[0,1]), ([-100.,-80.,20.,30.],[1,0])]:
    for e in ([[.01,0.,0.]], np.empty((0,3))):
        try:
            elastic_energy_density_2d(e, c, free)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonpositive free stiffness accepted')

# --- test case 13 ---
import numpy as np
# Component labels form a one-dimensional set of integer indices.
for free in ([1,1], [-1], [3], [.5], [[1]], None, [1+1j]):
    try:
        elastic_energy_density_2d([[.01,0.,0.]], [100.,80.,20.,30.], free)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid free component labels accepted')
for strain, stiffness in [([[.01+1j,0.,0.]],[100.,80.,20.,30.]), ([[.01,0.,0.]],[100.,80.,20+1j,30.])]:
    try:
        elastic_energy_density_2d(strain, stiffness)
    except ValueError:
        pass
    else:
        raise AssertionError('Complex physical input accepted')

# --- test case 14 ---
import numpy as np
# Positive free shear remains admissible far below ordinary stiffness scales.
r = elastic_energy_density_2d([[.01,.002,.03]], [100.,80.,20.,1e-12], [2])
assert np.allclose(r['relaxed_strains'], [[.01,.002,0.]], rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], [[1.04,.36,0.]], rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], .00556/16.02176634, rtol=1e-10, atol=1e-12)
# Almost singular but strictly positive free normal block is also admissible.
r = elastic_energy_density_2d([[.01,-.02,.003]], [100.,100.,99.999999,1e-6], [0,1])
assert np.allclose(r['relaxed_strains'], [[0.,0.,.003]], rtol=1e-9, atol=1e-10)
assert np.allclose(r['energy_density_ev_ang2'], .5e-6*.003**2/16.02176634, rtol=1e-10, atol=1e-12)

# --- test case 15 ---
import numpy as np
# Stationarity and energy minimality under admissible perturbations.
c = np.array([220.,75.,-24.,33.])
e = np.array([[.008,.02,.015],[-.005,-.01,.012]])
r = elastic_energy_density_2d(e, c, [1,2])
assert np.allclose(r['stress_npm'][:,[1,2]], 0., rtol=1e-9, atol=1e-8)
assert np.allclose(r['relaxed_strains'][:,0], e[:,0], rtol=1e-9, atol=1e-10)
# These trials alter only free components. The three returned fields have
# separate accuracy guarantees, so propagate their budgets into this identity.
true_free = np.column_stack((-c[2]/c[1]*e[:,0], np.zeros(len(e))))
strain_error = 1e-10 + 1e-9*np.abs(true_free)
true_energy = .5*(c[0]-c[2]**2/c[1])*e[:,0]**2/16.02176634
energy_error = 1e-12 + 1e-10*np.abs(true_energy)
for perturbation in ([.001,0.],[-.002,.003],[.003,-.002]):
    perturbation = np.asarray(perturbation)
    trial = r['relaxed_strains'].copy()
    trial[:,0] = e[:,0]
    trial[:,[1,2]] += perturbation
    trial_stress = np.column_stack((c[0]*trial[:,0]+c[2]*trial[:,1], c[2]*trial[:,0]+c[1]*trial[:,1], c[3]*trial[:,2]))
    trial_energy = .5*np.sum(trial*trial_stress,axis=1)/16.02176634
    expected_increase = (.5*c[1]*perturbation[0]**2+.5*c[3]*perturbation[1]**2)/16.02176634
    projection_error = np.sum(np.array([c[1],c[3]])*(
        np.abs(perturbation)*strain_error+.5*strain_error**2), axis=1)/16.02176634
    assert np.all(np.abs(trial_energy-r['energy_density_ev_ang2']-expected_increase)
                  <= energy_error+projection_error)
    assert np.all(trial_energy > r['energy_density_ev_ang2'])

# --- test case 16 ---
import numpy as np
# Formal harmonic equilibria can exceed the imposed-strain range.
r = elastic_energy_density_2d([[.01,.03,0.]], [100.,1.,9.,40.], [1])
assert np.allclose(r['relaxed_strains'], [[.01,-.09,0.]], rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], [[.19,0.,0.]], rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], .00095/16.02176634, rtol=1e-10, atol=1e-12)

# --- test case 17 ---
import numpy as np
# Free in-plane normal strains leave independently imposed engineering shear.
c = [120.,90.,-25.,35.]
r = elastic_energy_density_2d([[.018,-.02,.012],[-.012,.02,-.007]], c, [1,0])
assert np.allclose(r['relaxed_strains'], [[0.,0.,.012],[0.,0.,-.007]], rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], [[0.,0.,.42],[0.,0.,-.245]], rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], .5*35*np.array([.012**2,.007**2])/16.02176634, rtol=1e-10, atol=1e-12)

# --- test case 18 ---
import numpy as np
# Even an extremely small strictly positive free normal stiffness is a minimum.
r = elastic_energy_density_2d([[.03,.012,.01]], [1e-12,90.,0.,35.], [0])
assert np.allclose(r['relaxed_strains'], [[0.,.012,.01]], rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], [[0.,1.08,.35]], rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], (.5*90*.012**2+.5*35*.01**2)/16.02176634, rtol=1e-10, atol=1e-12)

# --- test case 19 ---
import numpy as np
# A singular normal block has no unique free equilibrium, even when rounded
# factorization pivots look weakly positive. A strictly positive neighbour does.
for coefficients in ([128.,128.,128.,32.], [2.,8.,4.,3.]):
    try:
        elastic_energy_density_2d([[.01,.02,0]], coefficients, [0,1])
    except ValueError:
        pass
    else:
        raise AssertionError("Singular free normal curvature was accepted")
coefficients=[128.,128.,128.-2**-20,32.]
r=elastic_energy_density_2d([[.01,.02,.01]], coefficients, [0,1])
assert np.allclose(r['relaxed_strains'],[[0.,0.,.01]],rtol=1e-9,atol=1e-10)
assert np.allclose(r['stress_npm'],[[0.,0.,.32]],rtol=1e-9,atol=1e-8)
assert np.isclose(r['energy_density_ev_ang2'][0],.0016/16.02176634,rtol=1e-10,atol=1e-12)

# --- test case 20 ---
import numpy as np
# Prescribed harmonic work remains accurate at the coefficient calibration limit.
c = np.array([10000., 6500., -1700., 2400.])
e = np.array([[.03, -.018, .012], [-.011, .023, -.027], [.008, .006, 0.]])
a, b, k, shear = c.astype(np.longdouble)
x, y, g = e.astype(np.longdouble).T
expected_stress = np.column_stack((a*x+k*y, k*x+b*y, shear*g))
expected_energy = (a*x*x/2+b*y*y/2+k*x*y+shear*g*g/2)/16.02176634
r = elastic_energy_density_2d(e, c)
assert np.allclose(r['relaxed_strains'], e, rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], expected_stress, rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], expected_energy, rtol=1e-10, atol=1e-12)

# --- test case 21 ---
import numpy as np
# At the calibrated soft-normal endpoint, zero normal loads still relax to zero.
# Inspect the supplied binary coefficients in extended arithmetic; their actual
# determinant ratio lies immediately above 1e-8, rather than being rounded to zero.
c = np.array([128., 128., 128.*np.sqrt(1.-1e-8), 32.])
a, b, k = c[:3].astype(np.longdouble)
ratio = (a*b-k*k)/(a*b)
assert np.longdouble('1e-8') <= ratio < np.longdouble('1.0000001e-8')
e = np.array([[.013, -.021, 0.], [-.017, .006, 0.]])
r = elastic_energy_density_2d(e, c, [0,1])
assert np.allclose(r['relaxed_strains'], 0., rtol=1e-9, atol=1e-10)
assert np.allclose(r['stress_npm'], 0., rtol=1e-9, atol=1e-8)
assert np.allclose(r['energy_density_ev_ang2'], 0., rtol=1e-10, atol=1e-12)

# --- test case 22 ---
import numpy as np
# An ignored free guess must still describe a real strain component.
strain=np.array([[.01, .02+1j, .003]],dtype=complex)
try:
    elastic_energy_density_2d(strain, [140.,100.,25.,35.], [1])
except ValueError:
    pass
else:
    raise AssertionError('A complex free strain guess was accepted')

# --- test case 23 ---
import numpy as np
# An empty batch does not waive the real numeric input-type contract.
strain=np.empty((0,3),dtype=complex)
try:
    elastic_energy_density_2d(strain, [140.,100.,25.,35.])
except ValueError:
    pass
else:
    raise AssertionError('An empty complex-typed strain batch was accepted')

# --- test case 24 ---
import numpy as np

# A free initial guess is still supplied physical input: validate it first.
try:
    elastic_energy_density_2d([[.01,.031,0.]], [140.,100.,25.,35.], [1])
except ValueError:
    pass
else:
    raise AssertionError('Out-of-range free strain guess was accepted')

# --- test case 25 ---
import numpy as np

# Missing or diverged physical records must not define an elastic equilibrium.
valid_strains=np.array([[.01,-.02,.003]])
valid_stiffness=np.array([140.,100.,25.,35.])
for invalid in (np.nan,np.inf,-np.inf):
    for free in ((),(1,)):
        strains=valid_strains.copy()
        strains[0,1]=invalid
        try:
            elastic_energy_density_2d(strains,valid_stiffness,free)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite prescribed or free strain was accepted')
    stiffness=valid_stiffness.copy()
    stiffness[2]=invalid
    try:
        elastic_energy_density_2d(valid_strains,stiffness)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite stiffness was accepted')
    try:
        elastic_energy_density_2d(valid_strains,valid_stiffness,[invalid])
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite free component label was accepted')

# --- test case 26 ---
import numpy as np

# A unique, unloaded normal equilibrium exists even below a 1e-9 determinant
# ratio. Zero normal forcing avoids amplification of a forcing-roundoff error.
stiffness=np.array([128.,128.,128.-2.**(-34),32.])
a,b,k=stiffness[:3].astype(np.longdouble)
assert 8e-13 < (a*b-k*k)/(a*b) < 1.1e-12
result=elastic_energy_density_2d([[.01,-.02,.003]],stiffness,[0,1])
assert np.allclose(result['relaxed_strains'],[[0.,0.,.003]],rtol=1e-9,atol=1e-10)
assert np.allclose(result['stress_npm'],[[0.,0.,.096]],rtol=1e-9,atol=1e-8)
assert np.allclose(result['energy_density_ev_ang2'],
                   [.5*32*.003**2/16.02176634],rtol=1e-10,atol=1e-12)

# The represented-input principal minor of this ordinary-sized tensor is
# positive (exact-rational ratio about 1.08181e-20), though product subtraction
# can round it to zero. With unloaded free normals its equilibrium stays zero.
stiffness=np.array([1.69443407281501,1.3988569383717917,1.5395684003547392,.5])
result=elastic_energy_density_2d([[.01,-.02,.003]],stiffness,[0,1])
assert np.allclose(result['relaxed_strains'],[[0.,0.,.003]],rtol=1e-9,atol=1e-10)
assert np.allclose(result['stress_npm'],[[0.,0.,.0015]],rtol=1e-9,atol=1e-8)
assert np.allclose(result['energy_density_ev_ang2'],
                   [.5*.5*.003**2/16.02176634],rtol=1e-10,atol=1e-12)

# A second strictly positive block has a zero rounded binary64 LU pivot.
# The validated, unforced minimum is zero regardless of that factorization.
stiffness=np.array([1.1,2.0454545454545454,1.5,.5])
result=elastic_energy_density_2d([[.01,-.02,.003]],stiffness,[0,1])
assert np.allclose(result['relaxed_strains'],[[0.,0.,.003]],rtol=1e-9,atol=1e-10)
assert np.allclose(result['stress_npm'],[[0.,0.,.0015]],rtol=1e-9,atol=1e-8)
assert np.allclose(result['energy_density_ev_ang2'],
                   [.5*.5*.003**2/16.02176634],rtol=1e-10,atol=1e-12)

# --- test case 27 ---
import numpy as np

# Empty observation batches do not waive material or component-type validation.
empty=np.empty((0,3))
stiffness=np.array([140.,100.,25.,35.])
invalid=[]
for bad in (np.nan,np.inf,-np.inf):
    coefficients=stiffness.copy()
    coefficients[2]=bad
    invalid.append((coefficients,()))
invalid.extend([(stiffness,np.empty(0,dtype=complex)),
                (stiffness.astype(complex),())])
for coefficients,free in invalid:
    try:
        elastic_energy_density_2d(empty,coefficients,free)
    except ValueError:
        pass
    else:
        raise AssertionError('Empty batch bypassed material or type validation')