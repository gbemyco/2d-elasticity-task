
## Scientific Background

This task examines the in-plane mechanical properties of two-dimensional (2D) materials using harmonic elasticity theory and DFT energy–strain data.

In DFT, 2D materials are modelled using supercells containing vacuum. To avoid dependence on vacuum thickness, elastic stiffness is expressed in N/m rather than the conventional 3D unit of Pa.

For an orthotropic 2D material, the engineering strain and stress vectors are:

    e = [exx, eyy, gxy], where gxy = 2*exy
    s = [sxx, syy, txy]

The elastic stiffness matrix is:

    C = [[C11, C12,   0],
         [C12, C22,   0],
         [  0,   0, C66]]

The harmonic elastic energy density is:

    U = 0.5 * eᵀ C e

DFT energy differences, normalized by the unstrained in-plane area, allow the four independent elastic constants to be determined using uniaxial, biaxial, and shear strains.

The compliance matrix, S = C⁻¹, describes the directional mechanical response. Young's modulus and Poisson's ratio are obtained by projecting the strain tensor along the loading and transverse directions, using the correct engineering-shear convention.

The final analysis reports continuous directional Young-modulus extrema over
all orientations, while its directional arrays are evaluated at the requested
angle samples. A sparse angle grid can miss an oblique soft or stiff direction.
The visible step prompt defines the canonical quadrant and numerical tie rule.

An almost-soft normal mode amplifies small coefficient errors, particularly in
the transverse response. The integrated result therefore distinguishes fitting
accuracy, arithmetic accuracy at a fitted tensor, and the precision with which
that tensor is reported. Step 2 constructs the regression features algebraically;
using a tolerance-limited energy helper would introduce an avoidable design error.

For reporting perturbations, scale the positive stiffness by
`diag(sqrt(C11), sqrt(C22), sqrt(C66))`. Its normalized normal block has eigenvalues
`1 +/- r`, where `r=abs(C12)/sqrt(C11*C22)`. A componentwise relative uncertainty
`eta` has normalized matrix norm at most `eta*(1+r)`. Dividing by the smallest
normalized eigenvalue gives the relative matrix perturbation bound `rho` stated
in Step 4. For `rho<1`, positive matrix ordering bounds each Young modulus,
and hence its global extrema, by the factors `1-rho` and `1+rho`.

The inverse matrix perturbation has relative norm at most `rho/(1-rho)`.
Cauchy-Schwarz for compliance work bounds the transverse response using the
longitudinal work for loading and perpendicular unit stresses. This gives the
Poisson uncertainty in the visible contract and the ratio bound for anisotropy.
It is a propagated reporting-error budget; strict mechanical admissibility and
the canonical extremum tie rule still apply at the reported fitted tensor.
Extremum directions and values have independent accuracy requirements: evaluating
a sharply peaked modulus at an approximately reported angle would mix those budgets.

