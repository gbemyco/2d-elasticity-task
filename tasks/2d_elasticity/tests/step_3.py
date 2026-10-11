
# --- test case 0 ---
import numpy as np

# Isotropic elasticity: rotational invariance
E0, nu0 = 200.0, 0.25
C = np.array([E0/(1-nu0**2), E0/(1-nu0**2),
              nu0*E0/(1-nu0**2), E0/(2*(1+nu0))])

angles = np.array([0, 0.17, np.pi/4, 1.1, np.pi/2, np.pi])
E, nu = directional_moduli_2d(C, angles)

assert np.allclose(E, E0, rtol=1e-9, atol=1e-8)
assert np.allclose(nu, nu0, rtol=1e-9, atol=1e-10)


# --- test case 1 ---
import numpy as np

# Anisotropic directional Young's modulus and Poisson ratio
C = np.array([160., 90., 25., 45.])
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

expected_E = [153.05555556, 126.78811122, 108.86937431,
              95.89557650, 86.09375]
expected_nu = [0.2777777777777778, 0.24065196548418022,
               0.2096597145993414, 0.18201595358955772, 0.15625]

E, nu = directional_moduli_2d(C, angles)

assert np.allclose(E, expected_E, rtol=1e-9, atol=1e-8)
assert np.allclose(nu, expected_nu, rtol=1e-9, atol=1e-10)


# --- test case 2 ---
import numpy as np

# Near-instability: small positive Born determinant
C = np.array([100., 80., 89., 20.])
angles = np.array([0, np.pi/4, np.pi/2])

expected_E = [0.9875, 53.10924369747895, 0.79]
expected_nu = [1.1125, 0.32773109243697296, 0.89]

E, nu = directional_moduli_2d(C, angles)

assert np.allclose(E, expected_E, rtol=1e-9, atol=1e-8)
assert np.allclose(nu, expected_nu, rtol=1e-9, atol=1e-10)


# --- test case 3 ---
import numpy as np

# Born-unstable stiffness must raise ValueError
C = np.array([100., 50., 80., 20.])

try:
    directional_moduli_2d(C, [0, np.pi/4])
except ValueError:
    pass
else:
    raise AssertionError("Unstable stiffness must raise ValueError")


# --- test case 4 ---
import numpy as np

# Auxetic isotropic material: negative Poisson ratio
E0, nu0 = 110.0, -0.30
C = np.array([E0/(1-nu0**2), E0/(1-nu0**2),
              nu0*E0/(1-nu0**2), E0/(2*(1+nu0))])

angles = np.array([0, 0.23, np.pi/4, 1.28, np.pi/2])
E, nu = directional_moduli_2d(C, angles)

assert E.shape == nu.shape == angles.shape
assert np.allclose(E, E0, rtol=1e-9, atol=1e-8)
assert np.allclose(nu, nu0, rtol=1e-9, atol=1e-10)

# --- test case 5 ---
import numpy as np

# Stiffness has four real coefficients in a one-dimensional vector.
for c in ([130., 85., 22.], [[130.,85.],[22.,48.]],
          [130.+1j,85.,22.,48.]):
    try:
        directional_moduli_2d(c, np.array([0.,np.pi/3]))
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid stiffness was accepted')

# --- test case 6 ---
import numpy as np

# Loading directions must be real nonempty one-dimensional samples.
C = [140., 90., 25., 50.]
for angles in (np.array([]), np.array([[0.,np.pi/2]]), np.array([.3+1j])):
    try:
        directional_moduli_2d(C, angles)
    except ValueError:
        pass
    else:
        raise AssertionError('Malformed angle samples were accepted')

# --- test case 7 ---
import numpy as np

# Born stability requires ALL positive normal and shear principal coefficients.
for c in ([-110.,-80.,20.,40.], [100.,70.,10.,0.], [100.,70.,10.,-1.], [100.,100.,100.,20.]):
    try:
        directional_moduli_2d(c, np.array([0.,np.pi/4]))
    except ValueError:
        pass
    else:
        raise AssertionError('Stiffness failed full Born positivity test')

# --- test case 8 ---
import numpy as np

# Almost-soft normal eigenmode: strict positive determinant is still stable.
C11, C22 = 120., 75.
C12 = np.sqrt(C11*C22*(1 - 1e-8))
C = np.array([C11, C22, C12, 40.])
assert 0 < C11*C22-C12*C12 < 0.001
angles = np.array([0.,np.pi/2])
E, nu = directional_moduli_2d(C, angles)
expected_E = np.array([C11-C12*C12/C22, C22-C12*C12/C11])
expected_nu = [C12/C22,C12/C11]
assert np.all(np.isfinite(E)) and np.all(np.isfinite(nu))
assert np.allclose(E, expected_E, rtol=1e-9, atol=1e-8)
assert np.allclose(nu, expected_nu, rtol=1e-9, atol=1e-10)

# --- test case 9 ---
import numpy as np

# Positive but very small shear modulus can dominate off-axis strain.
C = np.array([160., 90., 25., 1e-6])
E, nu = directional_moduli_2d(C, np.array([np.pi/4]))
det = C[0]*C[1]-C[2]**2
s11,s22,s12 = C[1]/det, C[0]/det, -C[2]/det
axial = (s11+s22+2*s12+1/C[3])/4
trans = (s11+s22+2*s12-1/C[3])/4
assert np.isclose(E[0], 1/axial, rtol=1e-9, atol=1e-8)
assert np.isclose(nu[0], -trans/axial, rtol=1e-9, atol=1e-10)

# --- test case 10 ---
import numpy as np
# Reflection, pi periodicity, and normal reciprocity are tensor identities.
C=np.array([260.,85.,-24.,31.]); a=np.array([-.37,.61,1.04,2.7])
E,nu=directional_moduli_2d(C,a)
Er,nur=directional_moduli_2d(C,-a)
Ep,nup=directional_moduli_2d(C,a+np.pi)
assert np.allclose(E,Er,rtol=2e-9,atol=2e-8)
assert np.allclose(E,Ep,rtol=2e-9,atol=2e-8)
assert np.allclose(nu,nur,rtol=2e-9,atol=2e-10)
assert np.allclose(nu,nup,rtol=2e-9,atol=2e-10)
Ex,nux=directional_moduli_2d(C,[0.,np.pi/2])
assert np.isclose(nux[0]/Ex[0],nux[1]/Ex[1],rtol=1e-9,atol=1e-10)

# --- test case 11 ---
import numpy as np
# Independently project the full strain tensor for mixed-sign coupling and large anisotropy.
C=np.array([920.,42.,-21.,8.]); a=np.array([-.9,.23,.83,2.2])
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
n=np.column_stack((np.cos(a),np.sin(a))); m=np.column_stack((-n[:,1],n[:,0]))
sigma=np.column_stack((n[:,0]**2,n[:,1]**2,n[:,0]*n[:,1]))
e=np.linalg.solve(matrix,sigma.T).T
T=np.zeros((len(a),2,2)); T[:,0,0]=e[:,0]; T[:,1,1]=e[:,1]
T[:,0,1]=T[:,1,0]=e[:,2]/2
axial=np.einsum('ni,nij,nj->n',n,T,n)
transverse=np.einsum('ni,nij,nj->n',m,T,m)
E,nu=directional_moduli_2d(C,a)
assert isinstance(E,np.ndarray) and isinstance(nu,np.ndarray)
assert E.shape==nu.shape==a.shape and E.dtype.kind==nu.dtype.kind=='f'
assert np.allclose(E,1/axial,rtol=1e-9,atol=1e-8)
assert np.allclose(nu,-transverse/axial,rtol=1e-9,atol=1e-10)

# --- test case 12 ---
import numpy as np
# 90-digit direct stress solves: almost-soft normal mode, well above floating zero.
C=np.array([1.0, 1.0, 0.9999999850988388, 0.5]); angles=np.array([0.0, 0.37, 1.5707963267948966])
E,nu=directional_moduli_2d(C,angles)
assert np.allclose(E,[2.9802322165650708e-08, 5.464948558629811e-08, 2.9802322165650708e-08],rtol=1e-9,atol=1e-8)
assert np.allclose(nu,[0.9999999850988388, 0.999999972675257, 0.9999999850988388],rtol=1e-9,atol=1e-10)

# --- test case 13 ---
import numpy as np
# Scale covariance: E scales with stiffness, while dimensionless Poisson does not.
C=np.array([155.,78.,23.,47.]); angles=np.array([-.2,.44,.91,3.2])
E,nu=directional_moduli_2d(C,angles)
E2,nu2=directional_moduli_2d(7*C,angles)
assert np.allclose(E2,7*E,rtol=2e-9,atol=8e-8)
assert np.allclose(nu2,nu,rtol=2e-9,atol=2e-10)

# --- test case 14 ---
import numpy as np

# A near-soft normal block can produce a sharp transverse-strain sign change
# near loading along its hard mode. Targets come from 90-digit direct stress
# solves, treating every displayed binary64 stiffness and angle as an exact input.
C = np.array([120., 75., 94.86832933070973, 40.])
angles = np.array([0.7266824343129394, 0.7267814343129394,
                   0.7267824343129394, 0.7267834343129395,
                   0.7268824343129394])
expected_E = np.array([19.636345424013943, 114.05252099579381,
                       114.10734006447952, 114.05241356461671,
                       19.635261230729075])
expected_nu = np.array([975.9633437502815, 57.03373047752011,
                        0.39860151172108726, -56.23590903717802,
                        -974.1394523567772])
young, poisson = directional_moduli_2d(C, angles)
assert np.allclose(young, expected_E, rtol=1e-9, atol=1e-8)
assert np.allclose(poisson, expected_nu, rtol=1e-9, atol=1e-10)

# --- test case 15 ---
import numpy as np

# Strong normal anisotropy and a near-soft determinant combine two sources of
# cancellation. The independent 90-digit stress solves use these exact inputs.
C = np.array([10000., 1., 99.99999949999999, 40.])
angles = np.array([0.09956865249116204, 0.09966765249116204,
                   0.09966865249116204, 0.09966965249116204,
                   0.09976865249116204])
expected_E = np.array([24.81215158709201, 2881.040807271918,
                       2914.571426489591, 2880.927377665924,
                       24.762655140906165])
expected_nu = np.array([49104.616215592396, 57030.91907384817,
                        -13.57285778559216, -57056.04774040655,
                        -49053.44939591083])
young, poisson = directional_moduli_2d(C, angles)
assert np.allclose(young, expected_E, rtol=1e-9, atol=1e-8)
assert np.allclose(poisson, expected_nu, rtol=1e-9, atol=1e-10)

# --- test case 16 ---
import numpy as np

# Strict admissibility extends below the calibrated accuracy range.
# Principal-axis observations remain finite in these very soft materials.
angles=np.array([0.,np.pi/2])
tensors=(np.array([1e-12,90.,0.,35.]),
         np.array([160.,1e-12,0.,35.]),
         np.array([160.,90.,25.,1e-12]),
         np.array([128.,128.,128.-2.**(-34),32.]))
for stiffness in tensors:
    a,b,k,s=stiffness.astype(np.longdouble)
    determinant=a*b-k*k
    assert min(a,b,s) > 0 and determinant > 0
    if k > 100:
        assert 8e-13 < determinant/(a*b) < 1.1e-12
    young,poisson=directional_moduli_2d(stiffness,angles)
    for values in (young,poisson):
        assert isinstance(values,np.ndarray)
        assert values.shape == angles.shape
        assert np.issubdtype(values.dtype,np.floating)
        assert np.all(np.isfinite(values))

# --- test case 17 ---
import numpy as np

# Material coefficients and loading directions must be completed physical data.
valid_stiffness=np.array([130.,85.,22.,48.])
valid_angles=np.array([.2,.7])
for invalid in (np.nan,np.inf,-np.inf):
    for component in (2,3):
        stiffness=valid_stiffness.copy()
        stiffness[component]=invalid
        try:
            directional_moduli_2d(stiffness,valid_angles)
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite stiffness was accepted')
    angles=valid_angles.copy()
    angles[1]=invalid
    try:
        directional_moduli_2d(valid_stiffness,angles)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite loading direction was accepted')

# --- test case 18 ---
import numpy as np

# Each diagonal must be finite; numerical types remain real even when an
# imaginary component would happen to be zero.
stiffness=np.array([130.,85.,22.,48.])
angles=np.array([.2,.7])
invalid=[]
for bad in (np.nan,np.inf,-np.inf):
    for component in (0,1):
        coefficients=stiffness.copy()
        coefficients[component]=bad
        invalid.append((coefficients,angles))
for imaginary in (0.,1.):
    coefficients=stiffness.astype(complex)
    coefficients[0]+=1j*imaginary
    directions=angles.astype(complex)
    directions[0]+=1j*imaginary
    invalid.extend([(coefficients,angles),(stiffness,directions)])
for coefficients,directions in invalid:
    try:
        directional_moduli_2d(coefficients,directions)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid material or loading data was accepted')

# --- test case 19 ---
import numpy as np

# Analytic principal and diagonal loading at the high-stiffness calibration
# boundary. The normal determinant ratio is 7/16, well away from a soft mode.
stiffness=[10000.,10000.,7500.,10000.]
angles=[0.,np.pi/4,np.pi/2]
response=directional_moduli_2d(stiffness,angles)
assert isinstance(response,tuple) and len(response) == 2
for values in response:
    assert isinstance(values,np.ndarray) and values.shape == (3,)
    assert np.issubdtype(values.dtype,np.floating)
assert np.allclose(response[0],[4375.,56000./3,4375.],rtol=1e-9,atol=1e-8)
assert np.allclose(response[1],[.75,-1./15,.75],rtol=1e-9,atol=1e-10)

# --- test case 20 ---
import numpy as np

# A near-boundary positive minor must not be mistaken for zero by rounded
# products. Exact-rational input arithmetic independently establishes its sign.
# Below the calibrated regime check admissibility and finite principal loading,
# without imposing an additional high-accuracy response guarantee.
stiffness=np.array([1.69443407281501,1.3988569383717917,1.5395684003547392,.5])
angles=np.array([0.,np.pi/2])
response=directional_moduli_2d(stiffness,angles)
assert isinstance(response,tuple) and len(response) == 2
for values in response:
    assert isinstance(values,np.ndarray) and values.shape == angles.shape
    assert np.issubdtype(values.dtype,np.floating)
    assert np.all(np.isfinite(values))