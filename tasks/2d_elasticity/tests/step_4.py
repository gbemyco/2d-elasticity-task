# --- test case 0 ---
import numpy as np

# Stable anisotropic material: stiffness and directional extrema
C = np.array([160., 90., 25., 45.])
e = np.array([
    [0.01,0,0], [-0.01,0,0], [0,0.015,0], [0,-0.015,0],
    [0.01,0.01,0], [0.01,-0.01,0], [0,0,0.02], [0,0,-0.02]
])

x, y, g = e.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
angles = np.linspace(0, np.pi, 13)
r = analyze_elastic_2d(e, 166.4*(X @ C)/16.02176634, 166.4, angles)

keys = {"stiffness_npm", "born_stable", "young_npm", "poisson",
        "young_min_npm", "young_max_npm", "anisotropy_ratio",
        "theta_young_min_rad", "theta_young_max_rad"}

assert set(r) == keys and r["born_stable"] is True
assert r["young_npm"].shape == r["poisson"].shape == angles.shape
for name in ("young_min_npm", "young_max_npm", "anisotropy_ratio", "theta_young_min_rad", "theta_young_max_rad"):
    assert type(r[name]) is float
assert np.allclose(r["stiffness_npm"], C, rtol=1e-8, atol=1e-5)
assert np.isclose(r["young_min_npm"], 86.09375, rtol=1e-5, atol=1e-4)
assert np.isclose(r["young_max_npm"], 153.05555555555554, rtol=1e-5, atol=1e-4)
assert np.isclose(r["anisotropy_ratio"], 153.05555555555554/86.09375, rtol=1e-5, atol=1e-5)
assert np.isclose(r["theta_young_min_rad"], np.pi/2, rtol=1e-7, atol=1e-7)
assert np.isclose(r["theta_young_max_rad"], 0, rtol=1e-7, atol=1e-7)

# --- test case 1 ---
import numpy as np

# Born-unstable material: directional outputs must be NaN
C = np.array([100., 50., 80., 20.])
e = np.array([
    [0.01,0,0], [-0.01,0,0], [0,0.01,0], [0,-0.01,0],
    [0.01,0.01,0], [0.01,-0.01,0], [0,0,0.02], [0,0,-0.02]
])

x, y, g = e.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
angles = np.array([0, np.pi/4, np.pi/2])
r = analyze_elastic_2d(e, 120*(X @ C)/16.02176634, 120, angles)

assert r["born_stable"] is False
assert np.allclose(r["stiffness_npm"], C, rtol=1e-8, atol=1e-5)
assert r["young_npm"].shape == r["poisson"].shape == angles.shape
assert np.all(np.isnan(r["young_npm"]))
assert np.all(np.isnan(r["poisson"]))

for key in ("young_min_npm", "young_max_npm", "anisotropy_ratio",
            "theta_young_min_rad", "theta_young_max_rad"):
    assert np.isnan(r[key])

# --- test case 2 ---
import numpy as np

# Auxetic orthotropic material: negative Poisson ratio
C = np.array([130., 70., -14., 50.])
e = np.array([
    [0.01,0,0], [-0.012,0,0], [0,0.011,0], [0,-0.013,0],
    [0.012,0.008,0], [0.012,-0.008,0],
    [0,0,0.019], [0,0,-0.017], [0.009,-0.011,0.013]
])

x, y, g = e.T
X = np.column_stack((x**2/2, y**2/2, x*y, g**2/2))
angles = np.array([0, np.pi/2])
r = analyze_elastic_2d(e, 156*(X @ C)/16.02176634, 156, angles)

Ex = C[0] - C[2]**2/C[1]
Ey = C[1] - C[2]**2/C[0]

assert r["born_stable"] is True
assert np.allclose(r["stiffness_npm"], C, rtol=1e-8, atol=1e-5)
assert np.allclose(r["young_npm"], [Ex, Ey], rtol=1e-5, atol=1e-4)
assert np.allclose(r["poisson"], [-0.2, -14/130], rtol=1e-5, atol=1e-6)
assert np.isclose(r["anisotropy_ratio"], Ex/Ey, rtol=1e-5, atol=1e-5)

# --- test case 3 ---
import numpy as np

# Integrated function must reject invalid angular sampling independently.
e = np.array([[.01,0,0],[-.01,0,0],[0,.012,0],[0,-.012,0],
              [.01,.01,0],[.01,-.01,0],[0,0,.02],[0,0,-.02]])
x,y,g = e.T
C = np.array([160.,90.,25.,45.]); X = np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
energies = 150*(X @ C)/16.02176634
for angles in (np.array([[0.,1.]]), np.array([])):
    try:
        analyze_elastic_2d(e, energies, 150., angles)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid angles were accepted by integration')
    unstable_energy=150*(X @ np.array([-160.,-90.,25.,45.]))/16.02176634
    try:
        analyze_elastic_2d(e, unstable_energy, 150., angles)
    except ValueError:
        pass
    else:
        raise AssertionError('Unstable branch bypassed angle validation')

# --- test case 4 ---
import numpy as np

# Negative C66 or negative normal diagonals trigger unstable NaN branch.
e = np.array([[.01,0,0],[-.01,0,0],[0,.012,0],[0,-.012,0],
              [.01,.01,0],[.01,-.01,0],[0,0,.02],[0,0,-.02]])
x,y,g = e.T
X = np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
angles = np.array([0.,np.pi/4,np.pi/2])
for C in (np.array([120.,90.,20.,-5.]), np.array([-120.,-90.,20.,45.])):
    r = analyze_elastic_2d(e, 150*(X @ C)/16.02176634, 150., angles)
    assert r['born_stable'] is False
    assert np.allclose(r['stiffness_npm'],C,rtol=1e-8,atol=1e-5)
    assert np.all(np.isnan(r['young_npm'])) and np.all(np.isnan(r['poisson']))
    for k in ('young_min_npm','young_max_npm','anisotropy_ratio',
              'theta_young_min_rad','theta_young_max_rad'):
        assert type(r[k]) is float and np.isnan(r[k])

# --- test case 5 ---
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

# --- test case 6 ---
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

# --- test case 7 ---
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
    assert np.isclose(r[k],r2[k],rtol=1e-8,atol=1e-8)
for k in ('theta_young_min_rad','theta_young_max_rad'):
    assert np.isclose(r[k],r2[k],rtol=1e-7,atol=1e-7)
assert r['young_npm'].shape==(1,) and r2['young_npm'].shape==angles2.shape
assert r['young_min_npm'] < min(r['young_npm'])-1

# --- test case 8 ---
import numpy as np
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
area=161.

# Isotropic source material: infer properties of the returned fitted tensor.
E0,nu0=180.,.2
C=np.array([E0/(1-nu0**2),E0/(1-nu0**2),nu0*E0/(1-nu0**2),E0/(2*(1+nu0))])
angles=np.array([.31,.89,2.2])
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=area*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
r=analyze_elastic_2d(e,energy,area,angles)

fitted = np.asarray(r['stiffness_npm'],dtype=float)
assert np.allclose(fitted,C,rtol=1e-8,atol=1e-5)
ld = fitted.astype(np.longdouble)
expected_stability = bool(min(ld[0],ld[1],ld[3])>0 and ld[0]*ld[1]-ld[2]*ld[2]>0)
assert r['born_stable'] is expected_stability
summaries = ('young_min_npm','young_max_npm','anisotropy_ratio',
             'theta_young_min_rad','theta_young_max_rad')
if not expected_stability:
    assert np.all(np.isnan(r['young_npm'])) and np.all(np.isnan(r['poisson']))
    assert all(type(r[k]) is float and np.isnan(r[k]) for k in summaries)
else:
    K = np.array([[fitted[0],fitted[2],0.],
                  [fitted[2],fitted[1],0.],[0.,0.,fitted[3]]])

    def stress_response(theta):
        co,si = np.cos(theta),np.sin(theta)
        strain = np.linalg.solve(K,np.array([co*co,si*si,co*si]))
        axial = strain[0]*co*co+strain[1]*si*si+strain[2]*co*si
        lateral = strain[0]*si*si+strain[1]*co*co-strain[2]*co*si
        return 1/axial,-lateral/axial

    def slope(theta):
        co,si = np.cos(theta),np.sin(theta)
        strain = np.linalg.solve(K,[co*co,si*si,co*si])
        return 2*(strain[1]-strain[0])+(co*co-si*si)/fitted[3]

    expected_young,expected_poisson = stress_response(angles)
    assert np.allclose(r['young_npm'],expected_young,rtol=1e-9,atol=1e-8)
    assert np.allclose(r['poisson'],expected_poisson,rtol=1e-9,atol=1e-10)
    critical = [0.,float(np.pi/2)]
    left,right = critical
    fl,fr = slope(left),slope(right)
    if fl*fr < 0:
        for _ in range(64):
            mid=(left+right)/2
            fm=slope(mid)
            if fm == 0:
                left=right=mid
                break
            if (fl<0)==(fm<0):
                left,fl=mid,fm
            else:
                right=mid
        critical.append((left+right)/2)
    critical=np.sort(critical)
    values,_=stress_response(critical)
    low,high=float(np.min(values)),float(np.max(values))
    angle_low=critical[np.flatnonzero(np.abs(values-low)<=1e-10*abs(low))[0]]
    angle_high=critical[np.flatnonzero(np.abs(values-high)<=1e-10*abs(high))[0]]
    assert np.allclose([r['young_min_npm'],r['young_max_npm']],[low,high],rtol=1e-9,atol=1e-8)
    assert np.isclose(r['anisotropy_ratio'],high/low,rtol=1e-8,atol=1e-8)
    assert np.allclose([r['theta_young_min_rad'],r['theta_young_max_rad']],
                       [angle_low,angle_high],rtol=1e-7,atol=1e-7)

# --- test case 9 ---
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
s11,s22,s12,s66=matrix[1,1]/np.linalg.det(matrix[:2,:2]),matrix[0,0]/np.linalg.det(matrix[:2,:2]),-matrix[0,1]/np.linalg.det(matrix[:2,:2]),1/matrix[2,2]
aa=s11+s22-2*s12-s66
bb=2*s12+s66-2*s22
candidates=[0.,1.]
if aa != 0 and 0 < -bb/(2*aa) < 1:
    candidates.append(-bb/(2*aa))
t=np.asarray(candidates)
values=1/(aa*t*t+bb*t+s22)
expected_ratio=max(values)/min(values)
assert np.isclose(r['anisotropy_ratio'],expected_ratio,rtol=1e-8,atol=1e-8)

# --- test case 10 ---
import numpy as np
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
area=161.
C=np.array([160.0, 90.0, 25.0, 1e-06]); angles=np.array([0.0, 0.7853981633974483, 1.5707963267948966])
matrix=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=area*.5*np.sum(e*(e@matrix),axis=1)/16.02176634
r=analyze_elastic_2d(e,energy,area,angles)

# Small positive source shear can change sign within the allowed fit error.
# Stability and subsequent properties therefore follow the returned fitted tensor.

fitted = np.asarray(r['stiffness_npm'],dtype=float)
assert np.allclose(fitted,C,rtol=1e-8,atol=1e-5)
ld = fitted.astype(np.longdouble)
expected_stability = bool(min(ld[0],ld[1],ld[3])>0 and ld[0]*ld[1]-ld[2]*ld[2]>0)
assert r['born_stable'] is expected_stability
summaries = ('young_min_npm','young_max_npm','anisotropy_ratio',
             'theta_young_min_rad','theta_young_max_rad')
if not expected_stability:
    assert np.all(np.isnan(r['young_npm'])) and np.all(np.isnan(r['poisson']))
    assert all(type(r[k]) is float and np.isnan(r[k]) for k in summaries)
else:
    K = np.array([[fitted[0],fitted[2],0.],
                  [fitted[2],fitted[1],0.],[0.,0.,fitted[3]]])

    def stress_response(theta):
        co,si = np.cos(theta),np.sin(theta)
        strain = np.linalg.solve(K,np.array([co*co,si*si,co*si]))
        axial = strain[0]*co*co+strain[1]*si*si+strain[2]*co*si
        lateral = strain[0]*si*si+strain[1]*co*co-strain[2]*co*si
        return 1/axial,-lateral/axial

    def slope(theta):
        co,si = np.cos(theta),np.sin(theta)
        strain = np.linalg.solve(K,[co*co,si*si,co*si])
        return 2*(strain[1]-strain[0])+(co*co-si*si)/fitted[3]

    expected_young,expected_poisson = stress_response(angles)
    assert np.allclose(r['young_npm'],expected_young,rtol=1e-9,atol=1e-8)
    assert np.allclose(r['poisson'],expected_poisson,rtol=1e-9,atol=1e-10)
    critical = [0.,float(np.pi/2)]
    left,right = critical
    fl,fr = slope(left),slope(right)
    if fl*fr < 0:
        for _ in range(64):
            mid=(left+right)/2
            fm=slope(mid)
            if fm == 0:
                left=right=mid
                break
            if (fl<0)==(fm<0):
                left,fl=mid,fm
            else:
                right=mid
        critical.append((left+right)/2)
    critical=np.sort(critical)
    values,_=stress_response(critical)
    low,high=float(np.min(values)),float(np.max(values))
    angle_low=critical[np.flatnonzero(np.abs(values-low)<=1e-10*abs(low))[0]]
    angle_high=critical[np.flatnonzero(np.abs(values-high)<=1e-10*abs(high))[0]]
    assert np.allclose([r['young_min_npm'],r['young_max_npm']],[low,high],rtol=1e-9,atol=1e-8)
    assert np.isclose(r['anisotropy_ratio'],high/low,rtol=1e-8,atol=1e-8)
    assert np.allclose([r['theta_young_min_rad'],r['theta_young_max_rad']],
                       [angle_low,angle_high],rtol=1e-7,atol=1e-7)

# --- test case 11 ---
import numpy as np
# Integration must honor fitting validation, not silently repair or broadcast inputs.
e=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],[.01,-.01,0],[0,0,.02]])
energy=np.ones(5); angles=np.array([.2,.7])
for strains,y,area in [(e,energy.reshape(5,1),120.),(e,energy[:4],120.),
                        (e,energy,0.),(e,energy,-120.),
                        (e[:3],energy[:3],120.)]:
    try:
        analyze_elastic_2d(strains,y,area,angles)
    except ValueError:
        pass
    else:
        raise AssertionError('Integrated fit accepted invalid data')

# --- test case 12 ---
import numpy as np
# Decoupled pure-shear observations identify a small negative curvature with a fit-error margin.
e=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],[.01,-.01,0],[0,0,.02]])
C=np.array([120.,90.,20.,-1e-3]); area=150.
x,y,g=e.T; X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
r=analyze_elastic_2d(e,area*(X@C)/16.02176634,area,np.array([.31,.8]))
assert r['born_stable'] is False
assert np.allclose(r['stiffness_npm'],C,rtol=1e-8,atol=1e-5)
assert np.all(np.isnan(r['young_npm'])) and np.all(np.isnan(r['poisson']))
for key in ('young_min_npm','young_max_npm','anisotropy_ratio','theta_young_min_rad','theta_young_max_rad'):
    assert type(r[key]) is float and np.isnan(r[key])


# --- test case 13 ---
import numpy as np
# Fit boundary data through the public API; classify the actual returned tensor.
# Zero-shear observations are independent of the normal regression. The second
# dataset has zero normal determinant and uses dyadic strains and exact units.
angles=np.array([.23,.81])
normal=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],[.01,-.01,0],[0,0,.02]])
dyadic=np.array([[5,3,7],[3,-4,2],[2,-6,1],[8,4,10],
                 [-7,-12,-10],[-7,8,9],[-6,-3,0],[-11,-12,5]])/512
for e,C in ((normal,np.array([120.,90.,20.,0.])),
            (dyadic,np.array([128.,128.,128.,32.]))):
    x,y,g=e.T
    X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
    r=analyze_elastic_2d(e,X@C,16.02176634,angles)
    fitted=np.asarray(r['stiffness_npm'],dtype=float)
    assert np.allclose(fitted,C,rtol=1e-8,atol=1e-5)
    a,b,k,shear=fitted.astype(np.longdouble)
    stable=bool(min(a,b,shear)>0 and a*b-k*k>0)
    assert r['born_stable'] is stable
    assert r['young_npm'].shape==r['poisson'].shape==angles.shape
    if not stable:
        assert np.all(np.isnan(r['young_npm'])) and np.all(np.isnan(r['poisson']))
        for key in ('young_min_npm','young_max_npm','anisotropy_ratio',
                    'theta_young_min_rad','theta_young_max_rad'):
            assert type(r[key]) is float and np.isnan(r[key])
    else:
        # An allowed numerical fit error can move an exact source boundary.
        # No source-boundary snapping or arbitrary positivity cutoff is required.
        assert np.all(np.isfinite(r['young_npm'])) and np.all(np.isfinite(r['poisson']))
        assert all(np.isfinite(r[key]) for key in ('young_min_npm','young_max_npm',
                   'anisotropy_ratio','theta_young_min_rad','theta_young_max_rad'))

# --- test case 14 ---
import numpy as np

# Infer a nearly soft normal mode; evaluate its physical response at the fit.
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
C=np.array([128.,128.,128.-2**-20,32.])
area=161.
K=np.array([[C[0],C[2],0.],[C[2],C[1],0.],[0.,0.,C[3]]])
energy=area*.5*np.sum(e*(e@K),axis=1)/16.02176634
angles=np.array([0.,.23,np.pi/4,1.19,np.pi/2])
r=analyze_elastic_2d(e,energy,area,angles)
fitted=np.asarray(r['stiffness_npm'],dtype=float)
assert np.allclose(fitted,C,rtol=1e-8,atol=1e-5)
a,b,k,shear=fitted.astype(np.longdouble)
D=a*b-k*k
stable=bool(min(a,b,shear)>0 and D>0)
assert r['born_stable'] is stable
if not stable:
    assert np.all(np.isnan(r['young_npm'])) and np.all(np.isnan(r['poisson']))
else:
    # Normalize the normal block by diag(sqrt(a),sqrt(b)). The matrix
    # reporting envelope is bounded relative to this positive stiffness.
    # Binary64 spacing is an internal comparison constant, not a supplied
    # physical input. Its role and value are stated in the visible prompt.
    # Published reporting budget plus 64 binary64 spacings at 1.
    delta=1.4210854715202004e-14
    eta=np.longdouble(1e-12)+delta
    correlation=abs(k)/np.sqrt(a*b)
    smallest=D/(a*b*(1+correlation))
    rho=float(max(eta,eta*(1+correlation)/smallest))
    assert rho < 1, 'Reporting envelope must remain positive definite'
    # Independent extended-precision elimination for longitudinal and
    # transverse unit stresses supplies a compliance perturbation bound.
    theta=angles.astype(np.longdouble)
    co,si=np.cos(theta),np.sin(theta)
    u,v=co*co,si*si
    ratio=k/a
    pivot=D/a
    yy=(v-ratio*u)/pivot
    xx=(u-k*yy)/a
    gg=co*si/shear
    axial=xx*u+yy*v+gg*co*si
    transverse=xx*v+yy*u-gg*co*si
    my=(u-ratio*v)/pivot
    mx=(v-k*my)/a
    transverse_work=mx*v+my*u+u*v/shear
    expected_E,expected_nu=1/axial,-transverse/axial
    E_budget=1e-8+(1e-9+rho)*abs(expected_E)
    nu_budget=(1e-10+1e-9*abs(expected_nu)
               +(1+rho)*rho/(1-rho)*(np.sqrt(transverse_work/axial)+abs(expected_nu)))
    assert np.all(abs(r['young_npm']-expected_E)<=E_budget)
    assert np.all(abs(r['poisson']-expected_nu)<=nu_budget)
    assert all(np.isfinite(r[key]) for key in ('young_min_npm','young_max_npm','anisotropy_ratio'))
    assert r['young_min_npm']>0

# --- test case 15 ---
import numpy as np

# A second identifiable design fits the same nearly soft normal mode.
e=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],[.01,-.01,0],[0,0,.02]])
C=np.array([128.,128.,128.-2**-20,32.])
x,y,g=e.T
X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
angles=np.array([0.,.23,np.pi/4,1.19,np.pi/2])
r=analyze_elastic_2d(e,X@C,16.02176634,angles)
fitted=np.asarray(r['stiffness_npm'],dtype=float)
assert np.allclose(fitted,C,rtol=1e-8,atol=1e-5)
a,b,k,shear=fitted.astype(np.longdouble)
D=a*b-k*k
stable=bool(min(a,b,shear)>0 and D>0)
assert r['born_stable'] is stable
if not stable:
    assert np.all(np.isnan(r['young_npm'])) and np.all(np.isnan(r['poisson']))
else:
    # Normalize the normal block by diag(sqrt(a),sqrt(b)). The matrix
    # reporting envelope is bounded relative to this positive stiffness.
    # Binary64 spacing is an internal comparison constant, not a supplied
    # physical input. Its role and value are stated in the visible prompt.
    # Published reporting budget plus 64 binary64 spacings at 1.
    delta=1.4210854715202004e-14
    eta=np.longdouble(1e-12)+delta
    correlation=abs(k)/np.sqrt(a*b)
    smallest=D/(a*b*(1+correlation))
    rho=float(max(eta,eta*(1+correlation)/smallest))
    assert rho < 1, 'Reporting envelope must remain positive definite'
    # Independent eigenmode stress work, with algebraic eigenvectors.
    hard=(a+b+np.hypot(a-b,2*k))/2
    soft=D/hard
    vx,vy=((hard-b,k) if a>=b else (k,hard-a))
    length=np.hypot(vx,vy)
    cosine,sine=vx/length,vy/length

    def response(theta):
        theta=np.asarray(theta,dtype=np.longdouble)
        u,v=np.cos(theta)**2,np.sin(theta)**2
        ph,ps=cosine*u+sine*v,-sine*u+cosine*v
        th,ts=cosine*v+sine*u,-sine*v+cosine*u
        axial=ph*ph/hard+ps*ps/soft+u*v/shear
        transverse=ph*th/hard+ps*ts/soft-u*v/shear
        transverse_work=th*th/hard+ts*ts/soft+u*v/shear
        return 1/axial,-transverse/axial,transverse_work/axial

    expected_E,expected_nu,work_ratio=response(angles)
    E_budget=1e-8+(1e-9+rho)*abs(expected_E)
    nu_budget=(1e-10+1e-9*abs(expected_nu)
               +(1+rho)*rho/(1-rho)*(np.sqrt(work_ratio)+abs(expected_nu)))
    assert np.all(abs(r['young_npm']-expected_E)<=E_budget)
    assert np.all(abs(r['poisson']-expected_nu)<=nu_budget)
    # Locate extrema independently. A permitted angle error must not be
    # converted into an extra modulus error at a sharply peaked maximum.
    def slope(theta):
        u,v=np.cos(np.longdouble(theta))**2,np.sin(np.longdouble(theta))**2
        ph,ps=cosine*u+sine*v,-sine*u+cosine*v
        ex=cosine*ph/hard-sine*ps/soft
        ey=sine*ph/hard+cosine*ps/soft
        return 2*(ey-ex)+(u-v)/shear

    critical=[0.,float(np.pi/2)]
    left,right=critical
    fl,fr=slope(left),slope(right)
    if fl*fr < 0:
        for _ in range(64):
            mid=(left+right)/2
            fm=slope(mid)
            if fm == 0:
                left=right=mid
                break
            if (fl<0)==(fm<0):
                left,fl=mid,fm
            else:
                right=mid
        critical.append((left+right)/2)
    critical=np.sort(critical)
    values,_,_=response(critical)
    low,high=np.min(values),np.max(values)
    expected_values=np.array([low,high])
    returned_values=np.array([r['young_min_npm'],r['young_max_npm']])
    assert np.all(abs(returned_values-expected_values)<=1e-8+(1e-9+rho)*abs(expected_values))
    expected_angles=[]
    for extremum in (low,high):
        tied=np.flatnonzero(abs(values-extremum)<=1e-10*abs(extremum))
        expected_angles.append(float(critical[tied[0]]))
    assert np.allclose([r['theta_young_min_rad'],r['theta_young_max_rad']],
                       expected_angles,rtol=1e-7,atol=1e-7)
    expected_ratio=high/low
    ratio_budget=1e-8+(1e-8+2*rho/(1-rho))*abs(expected_ratio)
    assert abs(r['anisotropy_ratio']-expected_ratio)<=ratio_budget

# --- test case 16 ---
import numpy as np
# Non-isotropic endpoint maxima: actual fitted gaps inside and outside
# the relative tie scale determine the canonical direction.
e=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],[.01,-.01,0],[0,0,.02]])
x,y,g=e.T
X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
for difference in (0.,.8e-10,1.2e-10):
    C=np.array([100.,100*(1+(1)*difference),20.,10.0])
    r=analyze_elastic_2d(e,X@C,16.02176634,np.array([.21,.97]))
    fitted=np.asarray(r['stiffness_npm'],dtype=float)
    assert np.allclose(fitted,C,rtol=1e-8,atol=1e-5)
    a,b,k,shear=fitted.astype(np.longdouble)
    D=a*b-k*k
    assert r['born_stable'] is True
    endpoints=np.array([D/b,D/a],dtype=np.longdouble)
    extremum=np.max(endpoints)
    tied=np.flatnonzero(np.abs(endpoints-extremum)<=1e-10*abs(extremum))
    expected_angle=float(np.array([0.,np.pi/2])[tied[0]])
    assert np.isclose(r['theta_young_max_rad'],expected_angle,rtol=1e-7,atol=1e-7)
    assert np.isclose(r['young_max_npm'],float(extremum),rtol=1e-9,atol=1e-8)

# --- test case 17 ---
import numpy as np
# Non-isotropic endpoint minima: actual fitted gaps inside and outside
# the relative tie scale determine the canonical direction.
e=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],[.01,-.01,0],[0,0,.02]])
x,y,g=e.T
X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
for difference in (0.,.8e-10,1.2e-10):
    C=np.array([100.,100*(1+(-1)*difference),20.,100.0])
    r=analyze_elastic_2d(e,X@C,16.02176634,np.array([.21,.97]))
    fitted=np.asarray(r['stiffness_npm'],dtype=float)
    assert np.allclose(fitted,C,rtol=1e-8,atol=1e-5)
    a,b,k,shear=fitted.astype(np.longdouble)
    D=a*b-k*k
    assert r['born_stable'] is True
    endpoints=np.array([D/b,D/a],dtype=np.longdouble)
    extremum=np.min(endpoints)
    tied=np.flatnonzero(np.abs(endpoints-extremum)<=1e-10*abs(extremum))
    expected_angle=float(np.array([0.,np.pi/2])[tied[0]])
    assert np.isclose(r['theta_young_min_rad'],expected_angle,rtol=1e-7,atol=1e-7)
    assert np.isclose(r['young_min_npm'],float(extremum),rtol=1e-9,atol=1e-8)

# --- test case 18 ---
import numpy as np

# Unequal normal stiffnesses and a nearly soft mode produce a sharp oblique
# response. Reporting precision must be propagated through that response.
source=np.array([1e4,1.,np.sqrt(1e4*(1-1e-8)),40.])
e=np.array([[.012,0,0],[-.017,0,0],[0,.014,0],[0,-.019,0],
            [.01,.01,0],[.01,-.01,0],[0,0,.022],[0,0,-.016],
            [.009,-.011,.013]])
x,y,g=e.T
X=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
alignment=np.arctan((source[1]/source[0])**.25)
angles=alignment+np.array([-1e-6,0.,1e-6])
r=analyze_elastic_2d(e,161*(X@source)/16.02176634,161.,angles)
fitted=np.asarray(r['stiffness_npm'],dtype=float)
assert np.allclose(fitted,source,rtol=1e-8,atol=1e-5)
a,b,k,shear=fitted.astype(np.longdouble)
D=a*b-k*k
stable=bool(min(a,b,shear)>0 and D>0)
assert r['born_stable'] is stable
if not stable:
    assert np.all(np.isnan(r['young_npm'])) and np.all(np.isnan(r['poisson']))
    assert all(np.isnan(r[key]) for key in ('young_min_npm','young_max_npm',
               'anisotropy_ratio','theta_young_min_rad','theta_young_max_rad'))
else:
    # Normalize the normal block by diag(sqrt(a),sqrt(b)). The matrix
    # reporting envelope is bounded relative to this positive stiffness.
    # Binary64 spacing is an internal comparison constant, not a supplied
    # physical input. Its role and value are stated in the visible prompt.
    # Published reporting budget plus 64 binary64 spacings at 1.
    delta=1.4210854715202004e-14
    eta=np.longdouble(1e-12)+delta
    correlation=abs(k)/np.sqrt(a*b)
    smallest=D/(a*b*(1+correlation))
    rho=float(max(eta,eta*(1+correlation)/smallest))
    assert rho < 1, 'Reporting envelope must remain positive definite'
    # Independent eigenmode stress work, with algebraic eigenvectors.
    hard=(a+b+np.hypot(a-b,2*k))/2
    soft=D/hard
    vx,vy=((hard-b,k) if a>=b else (k,hard-a))
    length=np.hypot(vx,vy)
    cosine,sine=vx/length,vy/length

    def response(theta):
        theta=np.asarray(theta,dtype=np.longdouble)
        u,v=np.cos(theta)**2,np.sin(theta)**2
        ph,ps=cosine*u+sine*v,-sine*u+cosine*v
        th,ts=cosine*v+sine*u,-sine*v+cosine*u
        axial=ph*ph/hard+ps*ps/soft+u*v/shear
        transverse=ph*th/hard+ps*ts/soft-u*v/shear
        transverse_work=th*th/hard+ts*ts/soft+u*v/shear
        return 1/axial,-transverse/axial,transverse_work/axial

    expected_E,expected_nu,work_ratio=response(angles)
    E_budget=1e-8+(1e-9+rho)*abs(expected_E)
    nu_budget=(1e-10+1e-9*abs(expected_nu)
               +(1+rho)*rho/(1-rho)*(np.sqrt(work_ratio)+abs(expected_nu)))
    assert np.all(abs(r['young_npm']-expected_E)<=E_budget)
    assert np.all(abs(r['poisson']-expected_nu)<=nu_budget)
    # Locate extrema independently. A permitted angle error must not be
    # converted into an extra modulus error at a sharply peaked maximum.
    def slope(theta):
        u,v=np.cos(np.longdouble(theta))**2,np.sin(np.longdouble(theta))**2
        ph,ps=cosine*u+sine*v,-sine*u+cosine*v
        ex=cosine*ph/hard-sine*ps/soft
        ey=sine*ph/hard+cosine*ps/soft
        return 2*(ey-ex)+(u-v)/shear

    critical=[0.,float(np.pi/2)]
    left,right=critical
    fl,fr=slope(left),slope(right)
    if fl*fr < 0:
        for _ in range(64):
            mid=(left+right)/2
            fm=slope(mid)
            if fm == 0:
                left=right=mid
                break
            if (fl<0)==(fm<0):
                left,fl=mid,fm
            else:
                right=mid
        critical.append((left+right)/2)
    critical=np.sort(critical)
    values,_,_=response(critical)
    low,high=np.min(values),np.max(values)
    expected_values=np.array([low,high])
    returned_values=np.array([r['young_min_npm'],r['young_max_npm']])
    assert np.all(abs(returned_values-expected_values)<=1e-8+(1e-9+rho)*abs(expected_values))
    expected_angles=[]
    for extremum in (low,high):
        tied=np.flatnonzero(abs(values-extremum)<=1e-10*abs(extremum))
        expected_angles.append(float(critical[tied[0]]))
    assert np.allclose([r['theta_young_min_rad'],r['theta_young_max_rad']],
                       expected_angles,rtol=1e-7,atol=1e-7)
    expected_ratio=high/low
    ratio_budget=1e-8+(1e-8+2*rho/(1-rho))*abs(expected_ratio)
    assert abs(r['anisotropy_ratio']-expected_ratio)<=ratio_budget

# --- test case 19 ---
import numpy as np

# A failed elastic fit's stability must not bypass real-angle validation.
e = np.array([[.01, 0, 0], [0, .012, 0], [.01, .01, 0],
              [.01, -.01, 0], [0, 0, .02]])
x, y, g = e.T
X = np.column_stack((x*x/2, y*y/2, x*y, g*g/2))
C = np.array([-120., -90., 20., 45.])
try:
    analyze_elastic_2d(e, X @ C, 16.02176634, np.array([.2+1j]))
except ValueError:
    pass
else:
    raise AssertionError('The unstable branch accepted complex angles')

# --- test case 20 ---
import numpy as np

# The isotropic energy dataset also exercises canonical ties. Independent
# critical-point work checks every returned model without an exact-fit branch.
e = np.array([[0, -10, 0], [2, 0, -10], [-4, -8, -6], [-10, 0, -1],
              [12, -5, 10], [-3, -3, -4], [11, 6, -8], [10, 11, 4]]) / 512.
x, y, g = e.T
X = np.column_stack((x*x/2, y*y/2, x*y, g*g/2))
C = np.array([96., 96., 48., 24.])
angles = np.array([.13, .69, 1.27])
r = analyze_elastic_2d(e, X @ C, 16.02176634, angles)
fitted = np.asarray(r['stiffness_npm'], dtype=float)
assert np.allclose(fitted, C, rtol=1e-8, atol=1e-5)
assert r['born_stable'] is True
assert r['young_npm'].shape == r['poisson'].shape == angles.shape

# Evaluate uniaxial-stress work independently from the fitted coefficients.
K = np.array([[fitted[0], fitted[2], 0.],
              [fitted[2], fitted[1], 0.], [0., 0., fitted[3]]])
def response(theta):
    co, si = np.cos(theta), np.sin(theta)
    strain = np.linalg.solve(K, np.array([co*co, si*si, co*si]))
    axial = strain[0]*co*co + strain[1]*si*si + strain[2]*co*si
    lateral = strain[0]*si*si + strain[1]*co*co - strain[2]*co*si
    return 1/axial, -lateral/axial

expected_young, expected_poisson = response(angles)
assert np.allclose(r['young_npm'], expected_young, rtol=1e-9, atol=1e-8)
assert np.allclose(r['poisson'], expected_poisson, rtol=1e-9, atol=1e-10)
# Locate all critical directions from independent unit-stress work. For
# isotropic work every critical value ties, so both targets are zero. For a
# legitimately asymmetric fit the same calculation checks its own canonical
# directions. Every returned angle and extremal value is checked on every run.
def slope(theta):
    co, si = np.cos(theta), np.sin(theta)
    strain = np.linalg.solve(K, [co*co, si*si, co*si])
    return 2*(strain[1]-strain[0]) + (co*co-si*si)/fitted[3]
critical = [0., float(np.pi/2)]
left, right = critical
fl, fr = slope(left), slope(right)
if fl*fr < 0:
    for _ in range(64):
        mid = (left+right)/2
        fm = slope(mid)
        if fm == 0:
            left = right = mid
            break
        if (fl < 0) == (fm < 0):
            left, fl = mid, fm
        else:
            right = mid
    critical.append((left+right)/2)
critical = np.sort(critical)
values, _ = response(critical)
low, high = float(np.min(values)), float(np.max(values))
angle_low = critical[np.flatnonzero(np.abs(values-low) <= 1e-10*abs(low))[0]]
angle_high = critical[np.flatnonzero(np.abs(values-high) <= 1e-10*abs(high))[0]]
assert np.allclose([r['young_min_npm'], r['young_max_npm']], [low, high],
                   rtol=1e-9, atol=1e-8)
assert np.isclose(r['anisotropy_ratio'], high/low, rtol=1e-8, atol=1e-8)
assert np.allclose([r['theta_young_min_rad'], r['theta_young_max_rad']],
                   [angle_low, angle_high], rtol=1e-7, atol=1e-7)

# --- test case 21 ---
import numpy as np
# Public dependency fixture: no function internals or evaluator introspection.
# The execution context must support rebinding this supplied public dependency.
strains = np.array([[.01,0.,0.],[0.,.012,0.],[.01,.01,0.],
                    [.01,-.01,0.],[0.,0.,.02]])
x,y,g = strains.T
design = np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
area = 150.
angles = np.array([0.,.23,np.pi/4,1.16,np.pi/2])

# Exact isotropy is prescribed at the returned fit, not inferred from energies.
stiffness = np.array([96.,96.,48.,24.])

energy = area*(design@stiffness)/16.02176634
original_fit = fit_stiffness_2d
fixture_calls = []
def exact_fit(strains, delta_energy_ev, area_ang2,
              tensor=stiffness, calls=fixture_calls):
    calls.append(True)
    return tensor.copy()
try:
    fit_stiffness_2d = exact_fit
    result = analyze_elastic_2d(strains,energy,area,angles)
finally:
    fit_stiffness_2d = original_fit
# Prove the fixture was used; do not silently fall back to a numerical fit.
assert fixture_calls, 'Public dependency fixture was not applied'
assert result['born_stable'] is True
assert np.allclose(result['stiffness_npm'],stiffness,rtol=1e-8,atol=1e-5)
for key in ('young_npm','poisson'):
    assert isinstance(result[key],np.ndarray) and result[key].shape == angles.shape
    assert result[key].dtype.kind == 'f' and np.all(np.isfinite(result[key]))
for key in ('young_min_npm','young_max_npm','anisotropy_ratio',
            'theta_young_min_rad','theta_young_max_rad'):
    assert type(result[key]) is float and np.isfinite(result[key])
assert np.all(result['young_npm'] > 0.)

# Isotropic constitutive work gives E=72 N/m, nu=1/2, and no preferred direction.
assert np.allclose(result['young_npm'],72.,rtol=1e-9,atol=1e-8)
assert np.allclose(result['poisson'],.5,rtol=1e-9,atol=1e-10)
assert np.allclose([result['young_min_npm'],result['young_max_npm']],
                   [72.,72.],rtol=1e-9,atol=1e-8)
assert np.isclose(result['anisotropy_ratio'],1.,rtol=1e-8,atol=1e-8)
# The independent target is zero for both directions; compare numerically
# using the published angle tolerance, as required by the standard guide.
assert np.allclose([result['theta_young_min_rad'],result['theta_young_max_rad']],
                   [0.0,0.0],rtol=1e-7,atol=1e-7)

# --- test case 22 ---
import numpy as np

# Prescribe returned tensors so a least-squares roundoff error cannot change
# the stability of a tiny positive modulus or nearly singular normal block.
strains=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                  [.01,-.01,0],[0,0,.02]])
x,y,g=strains.T
design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
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
    energy=150.*(design@stiffness)/16.02176634
    original_fit=fit_stiffness_2d
    fixture_calls=[]
    def exact_fit(strains,delta_energy_ev,area_ang2,
                  tensor=stiffness,calls=fixture_calls):
        calls.append(True)
        return tensor.copy()
    try:
        fit_stiffness_2d=exact_fit
        result=analyze_elastic_2d(strains,energy,150.,angles)
    finally:
        fit_stiffness_2d=original_fit
    assert fixture_calls, 'Public dependency fixture was not applied'
    assert result['born_stable'] is True
    assert np.allclose(result['stiffness_npm'],stiffness,rtol=1e-8,atol=1e-5)
    for key in ('young_npm','poisson'):
        values=result[key]
        assert isinstance(values,np.ndarray)
        assert values.shape == angles.shape
        assert np.issubdtype(values.dtype,np.floating)
        assert np.all(np.isfinite(values))
    for key in ('young_min_npm','young_max_npm','anisotropy_ratio',
                'theta_young_min_rad','theta_young_max_rad'):
        assert type(result[key]) is float and np.isfinite(result[key])

# --- test case 23 ---
import numpy as np

# Nonfinite loading directions remain invalid for a finite unstable fit.
strains=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                  [.01,-.01,0],[0,0,.02]])
x,y,g=strains.T
design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
stiffness=np.array([100.,50.,80.,20.])
assert stiffness[0]*stiffness[1]-stiffness[2]**2 < 0
assert np.linalg.matrix_rank(design) == 4
assert np.linalg.cond(design) < 1000
assert np.all(np.sqrt(np.mean(design*design,axis=0)) > 1e-6)
energy=150.*(design@stiffness)/16.02176634
for invalid in (np.nan,np.inf,-np.inf):
    try:
        analyze_elastic_2d(strains,energy,150.,np.array([.2,invalid]))
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite angles bypassed validation in unstable branch')

# --- test case 24 ---
import numpy as np
strains=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                  [.01,-.01,0],[0,0,.02]])
x,y,g=strains.T
design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert np.linalg.matrix_rank(design) == 4
assert np.linalg.cond(design) < 1000
assert np.all(np.sqrt(np.mean(design*design,axis=0)) > 1e-6)
area=150.

# Both stable and unstable source materials require genuine physical records.
angles=np.array([.2,.7])
for stiffness in (np.array([160.,90.,25.,45.]),
                  np.array([100.,50.,80.,20.])):
    energy=area*(design@stiffness)/16.02176634
    invalid=[]
    for bad in (np.nan,np.inf,-np.inf):
        observations=strains.copy()
        observations[0,0]=bad
        energies=energy.copy()
        energies[0]=bad
        directions=angles.copy()
        directions[0]=bad
        invalid.extend([(observations,energy,area,angles),
                        (strains,energies,area,angles),
                        (strains,energy,bad,angles),
                        (strains,energy,area,directions)])
    for imaginary in (0.,1.):
        observations=strains.astype(complex)
        observations[0,0]+=1j*imaginary
        energies=energy.astype(complex)
        energies[0]+=1j*imaginary
        directions=angles.astype(complex)
        directions[0]+=1j*imaginary
        invalid.extend([(observations,energy,area,angles),
                        (strains,energies,area,angles),
                        (strains,energy,complex(area,imaginary),angles),
                        (strains,energy,area,directions)])
    for observations,energies,basal_area,directions in invalid:
        try:
            analyze_elastic_2d(observations,energies,basal_area,directions)
        except ValueError:
            pass
        else:
            raise AssertionError('Integrated analysis accepted invalid physical records')

# --- test case 25 ---
import numpy as np
strains=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                  [.01,-.01,0],[0,0,.02]])
x,y,g=strains.T
design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
assert np.linalg.matrix_rank(design) == 4
assert np.linalg.cond(design) < 1000
assert np.all(np.sqrt(np.mean(design*design,axis=0)) > 1e-6)
area=150.

# A well-conditioned high-modulus sheet has tied principal minima and a true
# diagonal maximum. Fit-list inputs and sampled responses share the same units.
stiffness=np.array([10000.,10000.,7500.,10000.])
energy=area*(design@stiffness)/16.02176634
angles=[0.,np.pi/4,np.pi/2]
result=analyze_elastic_2d(strains.tolist(),energy.tolist(),area,angles)
assert result['born_stable'] is True
assert np.allclose(result['stiffness_npm'],stiffness,rtol=1e-8,atol=1e-5)
# Evaluate directions at the returned fit, so permissible fitting errors do not
# impose an unsupported source-material response accuracy downstream.
a,b,k,s=np.asarray(result['stiffness_npm'],dtype=np.longdouble)
determinant=a*b-k*k
u=np.cos(np.asarray(angles,dtype=np.longdouble))**2
v=np.sin(np.asarray(angles,dtype=np.longdouble))**2
axial=(b*u*u+a*v*v-2*k*u*v)/determinant+u*v/s
transverse=(a+b)*u*v/determinant-k*(u*u+v*v)/determinant-u*v/s
assert np.allclose(result['young_npm'],1/axial,rtol=1e-9,atol=1e-8)
assert np.allclose(result['poisson'],-transverse/axial,rtol=1e-9,atol=1e-10)
# Quadrant endpoints and the stationary compliance extremum determine the
# continuous extrema independently of the requested angle grid.
q0=b/determinant
q1=a/determinant
quadratic=(a+b+2*k)/determinant-1/slinear=-2*(a+k)/determinant+1/s
stationary=-linear/(2*quadratic)
assert 0 < stationary < 1
critical=np.array([0.,np.arccos(np.sqrt(stationary)),np.pi/2],dtype=np.longdouble)
t=np.cos(critical)**2
work=(quadratic*t+linear)*t+q1
values=1/work
low,high=min(values),max(values)
minimum=critical[np.flatnonzero(abs(values-low) <= 1e-10*abs(low))[0]]
maximum=critical[np.flatnonzero(abs(values-high) <= 1e-10*abs(high))[0]]
assert np.isclose(result['young_min_npm'],low,rtol=1e-9,atol=1e-8)
assert np.isclose(result['young_max_npm'],high,rtol=1e-9,atol=1e-8)
assert np.isclose(result['anisotropy_ratio'],high/low,rtol=1e-8,atol=1e-8)
assert np.allclose([result['theta_young_min_rad'],result['theta_young_max_rad']],
                   [minimum,maximum],rtol=1e-7,atol=1e-7)

# --- test case 26 ---
import numpy as np

# Classify the exactly returned tensor, independently of energy-fitting noise,
# when product rounding alone would erase a positive normal principal minor.
strains=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                  [.01,-.01,0],[0,0,.02]])
x,y,g=strains.T
design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
stiffness=np.array([1.69443407281501,1.3988569383717917,1.5395684003547392,.5])
energy=150.*(design@stiffness)/16.02176634
angles=np.array([0.,np.pi/2])
original_fit=fit_stiffness_2d
fixture_calls=[]
def exact_fit(strains,delta_energy_ev,area_ang2,
              tensor=stiffness,calls=fixture_calls):
    calls.append(True)
    return tensor.copy()
try:
    fit_stiffness_2d=exact_fit
    result=analyze_elastic_2d(strains,energy,150.,angles)
finally:
    fit_stiffness_2d=original_fit
assert fixture_calls, 'Public dependency fixture was not applied'
assert result['born_stable'] is True
assert np.allclose(result['stiffness_npm'],stiffness,rtol=1e-8,atol=1e-5)
for key in ('young_npm','poisson'):
    values=result[key]
    assert isinstance(values,np.ndarray) and values.shape == angles.shape
    assert np.issubdtype(values.dtype,np.floating)
    assert np.all(np.isfinite(values))
for key in ('young_min_npm','young_max_npm','anisotropy_ratio',
            'theta_young_min_rad','theta_young_max_rad'):
    assert type(result[key]) is float and np.isfinite(result[key])

# --- test case 27 ---
import numpy as np

# The integrated calculation uses its public directional response dependency
# for stable fits, and avoids undefined uniaxial response for unstable fits.
strains=np.array([[.01,0,0],[0,.012,0],[.01,.01,0],
                  [.01,-.01,0],[0,0,.02]])
x,y,g=strains.T
design=np.column_stack((x*x/2,y*y/2,x*y,g*g/2))
angles=np.array([.23,.81])
for stiffness,expected_stability in ((np.array([200.,100.,20.,40.]),True),
                                     (np.array([100.,50.,80.,20.]),False)):
    energy=150.*(design@stiffness)/16.02176634
    original_directional=directional_moduli_2d
    dependency_calls=[]
    def tensor_response(coefficients,directions,calls=dependency_calls):
        calls.append(True)
        a,b,k,s=np.asarray(coefficients,dtype=float)
        assert min(a,b,s) > 0 and a*b-k*k > 0
        co,si=np.cos(directions),np.sin(directions)
        stress=np.array([co*co,si*si,co*si])
        matrix=np.array([[a,k,0.],[k,b,0.],[0.,0.,s]])
        strain=np.linalg.solve(matrix,stress).T
        axial=strain[:,0]*co*co+strain[:,1]*si*si+strain[:,2]*co*si
        transverse=strain[:,0]*si*si+strain[:,1]*co*co-strain[:,2]*co*si
        return 1/axial,-transverse/axial
    try:
        directional_moduli_2d=tensor_response
        result=analyze_elastic_2d(strains,energy,150.,angles)
    finally:
        directional_moduli_2d=original_directional
    assert result['born_stable'] is expected_stability
    assert bool(dependency_calls) is expected_stability
    if expected_stability:
        expected_young,expected_poisson=tensor_response(result['stiffness_npm'],angles)
        assert np.allclose(result['young_npm'],expected_young,rtol=1e-9,atol=1e-8)
        assert np.allclose(result['poisson'],expected_poisson,rtol=1e-9,atol=1e-10)
    else:
        assert np.all(np.isnan(result['young_npm']))
        assert np.all(np.isnan(result['poisson']))