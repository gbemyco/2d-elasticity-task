
# --- test case 0 ---
import numpy as np

# Isotropic elasticity: analytical Young's modulus and Poisson ratio
E, nu = 180.0, 0.20
C = np.array([E/(1-nu**2), E/(1-nu**2),
              nu*E/(1-nu**2), E/(2*(1+nu))])

strains = np.array([
    [0.012,0,0], [-0.012,0,0], [0,0.014,0], [0,-0.014,0],
    [0.01,0.01,0], [0.01,-0.01,0],
    [0,0,0.018], [0,0,-0.018], [0.008,-0.006,0.01]
])

area = 164.0
x, y, g = strains.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
energies = area * (X @ C) / 16.02176634
angles = np.array([0, 0.31, np.pi/4, 1.22, np.pi/2, np.pi])

r = analyze_elastic_2d(strains, energies, area, angles)

assert r["born_stable"] is True
assert np.allclose(r["stiffness_npm"], C, rtol=1e-8, atol=1e-5)
assert np.allclose(r["young_npm"], E, rtol=1e-5, atol=1e-4)
assert np.allclose(r["poisson"], nu, rtol=1e-5, atol=1e-6)
assert np.isclose(r["anisotropy_ratio"], 1.0, rtol=1e-5, atol=1e-5)


# --- test case 1 ---
import numpy as np

# Anisotropic elasticity with deterministic DFT-like energy noise
C = np.array([160.0, 90.0, 25.0, 45.0])
strains = np.array([
    [0.01,0,0], [-0.01,0,0], [0,0.015,0], [0,-0.015,0],
    [0.01,0.01,0], [-0.01,-0.01,0], [0.01,-0.01,0],
    [0,0,0.02], [0,0,-0.02], [0.008,-0.012,0.01]
])

area = 166.4
x, y, g = strains.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
noise = 1e-6 * np.array([1,-1,0.5,-0.5,1.5,-1.5,0.75,-0.75,1.25,-0.25])
energies = area * (X @ C) / 16.02176634 + noise
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])

r = analyze_elastic_2d(strains, energies, area, angles)
Q, R = np.linalg.qr(X, mode="reduced")
expected = np.linalg.solve(R, Q.T @ (energies*16.02176634/area))

assert r["born_stable"] is True
assert np.allclose(r["stiffness_npm"], expected, rtol=1e-5, atol=1e-4)
assert np.isclose(r["young_min_npm"], min(r["young_npm"]), rtol=1e-5, atol=1e-4)
assert np.isclose(r["young_max_npm"], max(r["young_npm"]), rtol=1e-5, atol=1e-4)
assert r["anisotropy_ratio"] > 1.7


# --- test case 2 ---
import numpy as np

# Auxetic material: negative Poisson ratio and canonical principal-axis extrema
C = np.array([200.0, 80.0, -20.0, 60.0])
strains = np.array([
    [0.01,0,0], [-0.01,0,0], [0,0.015,0], [0,-0.015,0],
    [0.01,0.01,0], [0.01,-0.01,0],
    [0,0,0.02], [0,0,-0.02], [0.014,-0.009,0.012]
])

area = 168.0
x, y, g = strains.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
energies = area * (X @ C) / 16.02176634
angles = np.array([np.pi/2, 0, np.pi/2])

r = analyze_elastic_2d(strains, energies, area, angles)

assert r["born_stable"] is True
assert np.allclose(r["stiffness_npm"], C, rtol=1e-8, atol=1e-5)
assert np.allclose(r["young_npm"], [78,195,78], rtol=1e-5, atol=1e-4)
assert np.allclose(r["poisson"], [-0.1,-0.25,-0.1], rtol=1e-5, atol=1e-6)
assert np.isclose(r["anisotropy_ratio"], 2.5, rtol=1e-5, atol=1e-5)
assert np.isclose(r["theta_young_min_rad"], np.pi/2, rtol=1e-7, atol=1e-7)
assert np.isclose(r["theta_young_max_rad"], 0.0, rtol=1e-7, atol=1e-7)

# --- test case 3 ---
import numpy as np
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
area=161.
C=np.array([200.0, 80.0, 20.0, 10.0]); angles=np.array([0.0, 1.5707963267948966])
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=area*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
r=analyze_elastic_2d(e,energy,area,angles)

# Rational continuous-compliance extremum: cos(theta)^2=14/31.
assert r['born_stable'] is True
assert np.allclose(r['stiffness_npm'],C,rtol=1e-8,atol=1e-5)
assert np.isclose(r['young_min_npm'],34.44444444444444,rtol=1e-5,atol=1e-4)
assert np.isclose(r['young_max_npm'],195.0,rtol=1e-5,atol=1e-4)
assert np.isclose(r['theta_young_min_rad'],0.8338611064174528,rtol=1e-7,atol=1e-7)
assert np.isclose(r['theta_young_max_rad'],0.0,rtol=1e-7,atol=1e-7)
assert np.isclose(r['anisotropy_ratio'],5.661290322580645,rtol=1e-5,atol=1e-5)

# --- test case 4 ---
import numpy as np
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
area=161.
C=np.array([100.0, 70.0, 15.0, 200.0]); angles=np.array([0.0, 1.5707963267948966])
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=area*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
r=analyze_elastic_2d(e,energy,area,angles)

# Rational continuous-compliance extremum: cos(theta)^2=523/886.
assert r['born_stable'] is True
assert np.allclose(r['stiffness_npm'],C,rtol=1e-8,atol=1e-5)
assert np.isclose(r['young_min_npm'],67.75,rtol=1e-5,atol=1e-4)
assert np.isclose(r['young_max_npm'],160.87153881071268,rtol=1e-5,atol=1e-4)
assert np.isclose(r['theta_young_min_rad'],1.5707963267948966,rtol=1e-7,atol=1e-7)
assert np.isclose(r['theta_young_max_rad'],0.694606595062847,rtol=1e-7,atol=1e-7)
assert np.isclose(r['anisotropy_ratio'],2.3744876577226965,rtol=1e-5,atol=1e-5)

# --- test case 5 ---
import numpy as np
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
area=161.
C=np.array([200.0, 80.0, 20.0, 10.0]); angles=np.array([0.31])
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=area*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
r=analyze_elastic_2d(e,energy,area,angles)

# Global science cannot depend on one sparse, oblique angle sample.
angles2=np.array([1.2,0.,.83,-.41,2.7,.13,.13])
r2=analyze_elastic_2d(e,energy,area,angles2)
for k in ('young_min_npm','young_max_npm','anisotropy_ratio'):
    # Both independently computed results contribute to the comparison error.
    assert np.isclose(r[k],r2[k],rtol=2e-8,atol=2e-8)
for k in ('theta_young_min_rad','theta_young_max_rad'):
    assert np.isclose(r[k],r2[k],rtol=2e-7,atol=2e-7)
assert r['young_npm'].shape==(1,) and r2['young_npm'].shape==angles2.shape
assert r['young_min_npm'] < min(r['young_npm'])-1

# --- test case 6 ---
import numpy as np
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
area=161.

# Isotropic input data: fitted responses approach the analytic isotropic limit.
E0,nu0=180.,.2
C=np.array([E0/(1-nu0**2),E0/(1-nu0**2),nu0*E0/(1-nu0**2),E0/(2*(1+nu0))])
angles=np.array([.31,.89,2.2])
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=area*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
r=analyze_elastic_2d(e,energy,area,angles)
assert np.allclose(r['young_npm'],E0,rtol=1e-5,atol=1e-4)
assert np.allclose(r['poisson'],nu0,rtol=1e-5,atol=1e-6)
assert np.isclose(r['young_min_npm'],E0,rtol=1e-5,atol=1e-4)
assert np.isclose(r['young_max_npm'],E0,rtol=1e-5,atol=1e-4)
assert np.isclose(r['anisotropy_ratio'],1.,rtol=1e-5,atol=1e-5)

# Admissible fitting errors can break an exact source-tensor tie. Determine the
# canonical extrema at the returned tensor using independent direct stress solves.
fitted = r['stiffness_npm']
assert np.allclose(fitted,C,rtol=1e-8,atol=1e-5)
assert r['born_stable'] is True
matrix = np.array([[fitted[0], fitted[2], 0.],
                   [fitted[2], fitted[1], 0.], [0., 0., fitted[3]]])

def compliance_slope(theta):
    co, si = np.cos(theta), np.sin(theta)
    response = np.linalg.solve(matrix, [co*co, si*si, co*si])
    # Remove the known sin(2*theta) endpoint factor from d(1/E)/dtheta.
    return 2*(response[1]-response[0]) + (co*co-si*si)/fitted[3]

candidates = [0., float(np.pi/2)]
left, right = 0., float(np.pi/2)
fl, fr = compliance_slope(left), compliance_slope(right)
if (fl < 0 < fr) or (fr < 0 < fl):
    for _ in range(64):
        middle = (left+right)/2
        fm = compliance_slope(middle)
        if fm == 0:
            left = right = middle
            break
        if (fl < 0) == (fm < 0):
            left, fl = middle, fm
        else:
            right = middle
    candidates.append((left+right)/2)
candidates = np.sort(candidates)
co, si = np.cos(candidates), np.sin(candidates)
response = np.linalg.solve(matrix, np.array([co*co, si*si, co*si])).T
values = 1/(response[:,0]*co*co + response[:,1]*si*si + response[:,2]*co*si)
emin, emax = min(values), max(values)
amin = candidates[np.flatnonzero(abs(values-emin) <= 1e-10*abs(emin))[0]]
amax = candidates[np.flatnonzero(abs(values-emax) <= 1e-10*abs(emax))[0]]
assert np.isclose(r['young_min_npm'],emin,rtol=1e-9,atol=1e-8)
assert np.isclose(r['young_max_npm'],emax,rtol=1e-9,atol=1e-8)
assert np.isclose(r['theta_young_min_rad'],amin,rtol=1e-7,atol=1e-7)
assert np.isclose(r['theta_young_max_rad'],amax,rtol=1e-7,atol=1e-7)

# --- test case 7 ---
import numpy as np
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
area=161.
C=np.array([140.0, 95.0, -18.0, 24.0]); angles=np.array([-0.2, 0.43, 1.1, 2.8])
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=area*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
r=analyze_elastic_2d(e,energy,area,angles)

# Noise: validate at the returned fit, accounting for the whole error-propagation path.
energy=energy+1e-5*np.array([1,-1,.5,-.25,1.5,-.8,.2,-1.2,.3])
r=analyze_elastic_2d(e,energy,area,angles)
x,y,g=e.T; X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
Q,R=np.linalg.qr(X,mode='reduced')
target=np.linalg.solve(R,Q.T@(energy*16.02176634/area))
assert np.allclose(r['stiffness_npm'],target,rtol=1e-5,atol=1e-4)
fitted=r['stiffness_npm']
matrix=np.array([[fitted[0],fitted[2],0.],[fitted[2],fitted[1],0.],[0.,0.,fitted[3]]])
n=np.column_stack((np.cos(angles),np.sin(angles)))
stress=np.column_stack((n[:,0]**2,n[:,1]**2,n[:,0]*n[:,1]))
v=np.linalg.solve(matrix,stress.T).T
axial=v[:,0]*n[:,0]**2+v[:,1]*n[:,1]**2+v[:,2]*n[:,0]*n[:,1]
transverse=v[:,0]*n[:,1]**2+v[:,1]*n[:,0]**2-v[:,2]*n[:,0]*n[:,1]
assert np.allclose(r['young_npm'],1/axial,rtol=1e-9,atol=1e-8)
assert np.allclose(r['poisson'],-transverse/axial,rtol=1e-9,atol=1e-10)
critical=np.array([r['theta_young_min_rad'],r['theta_young_max_rad']])
co,si=np.cos(critical),np.sin(critical)
v=np.linalg.solve(matrix,np.array([co*co,si*si,co*si])).T
q=v[:,0]*co*co+v[:,1]*si*si+v[:,2]*co*si
assert np.allclose([r['young_min_npm'],r['young_max_npm']],1/q,rtol=1e-9,atol=1e-8)
dense=np.linspace(0,np.pi/2,3001)
co,si=np.cos(dense),np.sin(dense)
v=np.linalg.solve(matrix,np.array([co*co,si*si,co*si])).T
Edense=1/(v[:,0]*co*co+v[:,1]*si*si+v[:,2]*co*si)
assert r['young_min_npm'] <= min(Edense)+(1e-8+1e-9*abs(min(Edense)))
assert r['young_max_npm'] >= max(Edense)-(1e-8+1e-9*abs(max(Edense)))
assert np.isclose(r['anisotropy_ratio'],r['young_max_npm']/r['young_min_npm'],rtol=1e-8,atol=1e-8)

# --- test case 8 ---
import numpy as np
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
area=161.
C=np.array([160.0, 90.0, 25.0, 1e-06]); angles=np.array([0.0, 0.7853981633974483, 1.5707963267948966])
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=area*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
r=analyze_elastic_2d(e,energy,area,angles)

# A small source shear is below the permitted absolute fitting error. Stability
# and directional properties must therefore be checked at the actual fitted tensor.
fitted = r['stiffness_npm']
assert np.allclose(fitted,C,rtol=1e-8,atol=1e-5)
ld = fitted.astype(np.longdouble)
stable = bool(min(ld[0],ld[1],ld[3])>0 and ld[0]*ld[1]-ld[2]*ld[2]>0)
assert r['born_stable'] is stable
assert r['young_npm'].shape == r['poisson'].shape == angles.shape
if not stable:
    assert np.all(np.isnan(r['young_npm'])) and np.all(np.isnan(r['poisson']))
    for key in ('young_min_npm','young_max_npm','anisotropy_ratio',
                'theta_young_min_rad','theta_young_max_rad'):
        assert type(r[key]) is float and np.isnan(r[key])
else:
    matrix = np.array([[fitted[0], fitted[2], 0.],
                       [fitted[2], fitted[1], 0.], [0., 0., fitted[3]]])
    co, si = np.cos(angles), np.sin(angles)
    response = np.linalg.solve(matrix, np.array([co*co, si*si, co*si])).T
    axial = response[:,0]*co*co + response[:,1]*si*si + response[:,2]*co*si
    transverse = response[:,0]*si*si + response[:,1]*co*co - response[:,2]*co*si
    assert np.allclose(r['young_npm'],1/axial,rtol=1e-9,atol=1e-8)
    assert np.allclose(r['poisson'],-transverse/axial,rtol=1e-9,atol=1e-10)

    def compliance_slope(theta):
        co, si = np.cos(theta), np.sin(theta)
        response = np.linalg.solve(matrix, [co*co, si*si, co*si])
        return 2*(response[1]-response[0]) + (co*co-si*si)/fitted[3]

    candidates = [0., float(np.pi/2)]
    left, right = 0., float(np.pi/2)
    fl, fr = compliance_slope(left), compliance_slope(right)
    if (fl < 0 < fr) or (fr < 0 < fl):
        for _ in range(64):
            middle = (left+right)/2
            fm = compliance_slope(middle)
            if fm == 0:
                left = right = middle
                break
            if (fl < 0) == (fm < 0):
                left, fl = middle, fm
            else:
                right = middle
        candidates.append((left+right)/2)
    candidates = np.sort(candidates)
    co, si = np.cos(candidates), np.sin(candidates)
    response = np.linalg.solve(matrix, np.array([co*co, si*si, co*si])).T
    values = 1/(response[:,0]*co*co + response[:,1]*si*si + response[:,2]*co*si)
    emin, emax = min(values), max(values)
    amin = candidates[np.flatnonzero(abs(values-emin) <= 1e-10*abs(emin))[0]]
    amax = candidates[np.flatnonzero(abs(values-emax) <= 1e-10*abs(emax))[0]]
    assert np.isclose(r['young_min_npm'],emin,rtol=1e-9,atol=1e-8)
    assert np.isclose(r['young_max_npm'],emax,rtol=1e-9,atol=1e-8)
    assert np.isclose(r['anisotropy_ratio'],emax/emin,rtol=1e-8,atol=1e-8)
    assert np.isclose(r['theta_young_min_rad'],amin,rtol=1e-7,atol=1e-7)
    assert np.isclose(r['theta_young_max_rad'],amax,rtol=1e-7,atol=1e-7)

# --- test case 9 ---
import numpy as np
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
area=161.

# Valid energy datasets with independent failures of Born diagonal and determinant rules.
angles=np.array([.31,.83])
for C in (np.array([100.,50.,80.,20.]),np.array([-120.,-90.,20.,45.]),
          np.array([120.,90.,20.,-5.])):
    matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
    energy=area*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
    r=analyze_elastic_2d(e,energy,area,angles)
    assert r['born_stable'] is False
    assert np.allclose(r['stiffness_npm'],C,rtol=1e-8,atol=1e-5)
    assert r['young_npm'].shape==r['poisson'].shape==angles.shape
    assert np.all(np.isnan(r['young_npm'])) and np.all(np.isnan(r['poisson']))
    for k in ('young_min_npm','young_max_npm','anisotropy_ratio','theta_young_min_rad','theta_young_max_rad'):
        assert type(r[k]) is float and np.isnan(r[k])