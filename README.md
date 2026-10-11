# 2D elasticity task

A four-step scientific computation for an unstressed orthotropic monolayer,
using NumPy and Python built-ins only. All stiffnesses are areal quantities in
N/m; the model does not introduce a slab thickness.

1. Solve mixed prescribed-strain and zero-conjugate-stress boundary conditions,
   returning equilibrium engineering strain, stress, and stored energy.
2. Infer identifiable stiffness coefficients from energy observations, choosing
   a stable numerical method for the specified physical least-squares objective.
3. Project uniaxial loading through the engineering compliance tensor to obtain
   directional Young moduli and Poisson ratios.
4. Classify the fitted tensor and determine continuous global Young-modulus
   extrema and canonical loading directions, independently of the sample grid.

Steps 1 and 2 use the same harmonic work law. The freely relaxing components
in Step 1 are macroscopic strains, distinct from atomic-coordinate relaxation.
The energy dataset uses a consistent atomic relaxation convention. Steps 2 and
3 provide the public dependencies used by the integrated Step 4 calculation.

`tasks/2d_elasticity/problem.yaml` states the assumptions, input and output
contracts, units, numerical conventions, accuracy guarantees, and exceptions.
`source.md` preserves the supplied source and includes the elasticity paper.
Scaffolds are unimplemented; standalone references agree with `solution.py`.
`second_solution.py` uses independent constructions for the scientific work.

The task has 117 self-contained cases: 28, 30, 21, and 28 per step, plus 10
whole-task cases. Tests include analytic mechanics, inference with noise,
calibration endpoints, identifiable and unidentifiable strain designs, soft
normal and shear modes, auxetic responses, canonical ties, and invalid physical
records. NaN and infinities represent missing or diverged records and must raise
ValueError; finite unstable tensors have the specified NaN response outputs.

A principal-minor sign erased by rounded products is recomputed with exact
integer ratios of the represented coefficients. This chooses adequate arithmetic
precision rather than imposing a mechanical cutoff. The primary equilibrium
solver also handles validated zero forcing directly. These corrections leave
well-resolved mechanics and all numerical tolerance budgets unchanged.

Three Step 4 tests temporarily replace the supplied public fitting dependency;
one checks the public directional dependency in stable and unstable branches.
These tests verify invocation and require shared public dependency bindings. No graded test accesses function globals, reads source, unwraps an
evaluator proxy, or uses process, filesystem, or network tools.

Five scientific mutants are packaged: one per step and one whole-task mutant.
Author-only regression runners, additional controls, exact-arithmetic proofs,
and detailed records remain outside this archive. The accompanying report
maps scientific promises to tests and records local reference, independent,
no-op, numerical-noise, and fresh-process checks. Local validation does not
establish the submission platform's Oracle or content-quality approval.