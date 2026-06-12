# Usadel Dynamics

This repository contains numerical tools for homogeneous kinetic Usadel
dynamics in a BCS superconductor. The main solver lives in
`usadel_dynamics.py`, with a standalone clean-limit comparison helper in
`clean_BCS.py`.

## Module Overview

`usadel_dynamics.py` organizes the homogeneous kinetic Usadel workflow into a
small set of cooperating objects:

- `gridParams`: frequency and time grids.
- `modelParams`: physical model parameters and self-energy list.
- `solverParams`: numerical tolerances, iteration limits, and progress options.
- `SelfEnergy`: base interface for model self-energies.
- `VectorPotentialDrive`: time-dependent vector-potential self-energy.
- `UsadelAlgebra`: state encoding, Green functions, Hamiltonians, self-energies,
  and residual algebra.
- `EquilibriumModel`: self-consistent equilibrium gap and retarded-state solves.
- `BackwardEulerDynamics`: implicit backward-Euler time evolution.
- `StateBundle`: cached state representation containing `X`, `Delta`, `gR`,
  and `gK`.

`clean_BCS.py` is independent of the Usadel module and provides simple
clean-limit BCS current helper functions.

The encoded state is always the full per-frequency vector

```text
X_w = [Re chi_w, Im chi_w, f0_w, f3_w].
```

The spectral degree of freedom `chi_w` is algebraically constrained, but it is
kept in the state so equilibrium and dynamics use the same representation.

## Data Types and Shapes

The module uses NumPy arrays throughout. Nambu matrices are represented as
Pauli-vector components in the order

```text
[tau0, tau1, tau2, tau3].
```

Common shapes:

| Object | Type | Shape | Meaning |
| --- | --- | --- | --- |
| `ws` | real array | `(nw,)` | Frequency grid. |
| `time_points` | real array | `(nt,)` | Simulation times. |
| `dts` | real array | `(nt - 1,)` | Local time steps. |
| `chi` | complex array | `(nw,)` | Spectral angle. |
| `f` | complex array | `(nw,)` | Occupation encoding, with `real(f)=f0` and `imag(f)=f3`. |
| `X` | real array | `(4 * nw,)` | Flattened state vector. |
| `X.reshape((nw, 4))` | real array | `(nw, 4)` | Per-frequency `[Re chi, Im chi, f0, f3]`. |
| `gR`, `gK` | complex array | `(nw, 4)` | Retarded and Keldysh Green functions as Pauli vectors. |
| `gR_t_w`, `gK_t_w` | complex array | `(nt_returned, nw, 4)` | Retarded and Keldysh Green-function traces. |
| `hR`, `hK` | complex array | `(nw, 4)` | Retarded and Keldysh Hamiltonian/self-energy combinations. |
| `Delta` | float | scalar | Gap value associated with a state. |

`StateBundle` stores one time slice:

| Field | Type | Shape |
| --- | --- | --- |
| `X` | real array | `(4 * nw,)` |
| `Delta` | float | scalar |
| `gR` | complex array | `(nw, 4)` |
| `gK` | complex array | `(nw, 4)` |

`DynamicsResult` stores traces:

| Field | Type | Shape |
| --- | --- | --- |
| `times` | real array | `(nt_returned,)` |
| `X` | real array | `(nt_returned, 4 * nw)` |
| `Delta` | real array | `(nt_returned,)` |
| `converged` | bool array | `(nt_returned,)` |
| `scipy_converged` | bool array | `(nt_returned,)` |
| `residual_converged` | bool array | `(nt_returned,)` |
| `iterations` | int array | `(nt_returned,)` |
| `residual` | real array | `(nt_returned,)` |
| `step_time` | real array | `(nt_returned,)` |
| `current` | real array or `None` | `(nt_returned,)` |

If dynamics stops early after a failed step, `nt_returned` is shorter than the
input time grid.

## Basic Setup

```python
import numpy as np
import usadel_dynamics as ud

cutoff = 10.0
nw = 401
time_points = np.linspace(0.0, 10.0, 100)

grid = ud.gridParams(cutoff, nw, time_points)

Tc = 1.0
bcs = ud.guess_bcs_constant(cutoff, Tc)

model = ud.modelParams(
    grid=grid,
    eta=0.1,
    T=0.5,
    bcs_constant=bcs,
)

numerics = ud.solverParams(
    dynamics_tol=1.0e-7,
    dynamics_maxiter=2000,
)

algebra, equilibrium, dynamics = ud.build_default_models(model, numerics)
```

## Frequency and Time Grids

`gridParams` is responsible for constructing and storing the numerical grids.

```python
grid = ud.gridParams(cutoff=10.0, nw=401, time_points=np.linspace(0.0, 5.0, 200))
```

Derived fields:

- `grid.ws`: frequency grid generated from `cutoff` and `nw`.
- `grid.dw`: frequency spacing.
- `grid.time_points`: simulation times.
- `grid.nt`: number of time points.
- `grid.dts`: local time steps between adjacent time points.
- `grid.dt`: uniform time spacing if the grid is uniform, otherwise `nan`.
- `grid.is_uniform_time`: whether all local time steps are equal.
- `grid.dt_min` and `grid.dt_max`: minimum and maximum local time steps.

The time grid must be strictly increasing. Dynamics uses the local step
`t_new - t_old` at each backward-Euler update, so nonuniform time grids are
allowed.

## Model Parameters

`modelParams` stores physical parameters and self-energy contributions:

```python
model = ud.modelParams(
    grid=grid,
    eta=0.1,
    T=0.5,
    bcs_constant=bcs,
    self_energies=[],
)
```

`self_energies` may be:

- `None`, meaning no additional self-energy.
- A single `SelfEnergy` instance.
- A list or tuple of `SelfEnergy` instances.

All self-energies are summed before being subtracted from the bare Hamiltonian:

```text
h = h0 - sigma_total.
```

## Vector-Potential Drive

The vector potential is represented as a self-energy:

```python
pulse_center = 7.0
tau_p = 1.0
A0 = 0.7

A_t = lambda t, A0=A0, pulse_center=pulse_center, tau_p=tau_p: (
    A0 * np.exp(-0.5 * ((t - pulse_center) / tau_p) ** 2)
)

vp = ud.VectorPotentialDrive(A_t)

model = ud.modelParams(
    grid=grid,
    eta=0.1,
    T=0.5,
    bcs_constant=bcs,
    self_energies=[vp],
)
```

Use a distinct variable name such as `pulse_center` for the center of the pulse.
Avoid reusing that name for wall-clock timing, because Python lambdas capture
variables by reference.

For diagnostics, individual self-energy terms can be inspected:

```python
bundle = dynamics.initial_state
terms = algebra.sigma_terms("retarded", bundle.Delta, t=7.0, gR=bundle.gR)
print([np.max(np.abs(sigma)) for sigma in terms])
```

## Algebra Objects

`PauliAlgebra` contains low-level vectorized operations on Pauli-vector Nambu
matrices. Most users should not need to call it directly, but it defines the
matrix products, commutators, anticommutators, advanced components, and selected
component products used by the solver.

`UsadelAlgebra` is the main algebra interface. It owns the conversion between
the encoded state `X`, Green functions, Hamiltonians, and residuals:

```python
chi, f = algebra.components(X)
gR, gK = algebra.g_from_state(X)
gR_t_w, gK_t_w = algebra.g_from_state(result.X)
Delta = algebra.gap_from_state(X)
hR, hK = algebra.h_from_state(X, t)
bundle = algebra.bundle_state(X)
```

Common methods:

| Method | Purpose |
| --- | --- |
| `make_state(chi, f)` | Pack complex `chi` and occupation `f=f0+i f3` into flattened `X`. |
| `components(X)` | Unpack a flattened state or state trace into complex `chi` and `f`. |
| `occupation_components(X)` | Unpack a flattened state or state trace into real `f0` and `f3`. |
| `g_from_state(X)` | Construct `gR(X)` and `gK(X)` from a state or state trace. |
| `gR_from_state(X)` | Construct only `gR(X)` from a state or state trace. |
| `gK_from_state(X)` | Construct only `gK(X)` from a state or state trace. |
| `current_integrand_from_state(X, A)` | Compute the lowest-order Moyal current integrand for a state or trace with explicit `A`. |
| `current_from_state(X, A)` | Compute the lowest-order Moyal current `J/sigma_n` for a state or trace with explicit `A`. |
| `gap_from_state(X)` | Compute `Delta[X]` from `gK(X)`. |
| `gap_from_gK(gK)` | Compute the gap directly from a Keldysh Green function. |
| `h_from_state(X, t, Delta=None)` | Construct `hR`, `hK` using `Delta[X]` unless supplied. |
| `hamiltonian(Delta, t, gR, gK)` | Construct `h0 - sigma_total` for supplied Green functions. |
| `sigma(part, Delta, t, gR, gK)` | Sum all self-energy contributions for `"retarded"` or `"keldysh"`. |
| `sigma_terms(part, Delta, t, gR, gK)` | Return the individual self-energy terms before summing. |
| `bundle_state(X, Delta=None)` | Attach cached `gR`, `gK`, and `Delta` to an encoded state. |
| `backward_euler_residual_vector(...)` | Packed residual used by the nonlinear dynamics solver. |

The cleanest way to extract occupations from a dynamics result is through the
algebra helper:

```python
chi_t_w, f_t_w = algebra.components(result.X)
f0_t_w, f3_t_w = algebra.occupation_components(result.X)
f_t_w = f0_t_w + 1j * f3_t_w
gR_t_w, gK_t_w = algebra.g_from_state(result.X)
```

Here `f0_t_w[i, j]` and `f3_t_w[i, j]` are the occupation components at
`time=result.times[i]` and `frequency=grid.ws[j]`.

The lowest-order Moyal current is computed automatically in equilibrium and
dynamics results when `solverParams.compute_current=True`, the default.
Equilibrium uses the static/equilibrium vector potential from the model at
`t=0`; dynamics evaluates the vector potential on `result.times`.

```python
eq = equilibrium.equilibrium(Delta0=1.0)
J_eq = eq.current

result = dynamics.run()
J_t = result.current
```

Currents can also be recomputed from result objects:

```python
J_eq = equilibrium.current(eq)
J_t = dynamics.current(result)
```

The lower-level algebra method requires the vector potential to be supplied
explicitly:

```python
J_eq = algebra.current_from_state(eq.X, A=0.2)
J_t = algebra.current_from_state(result.X, A=A_t(result.times))
```

The returned current is normalized as `J/sigma_n`; the implementation currently
sets `sigma_n = 1`.

## Equilibrium

Equilibrium solves for the retarded state and self-consistent gap using the same
encoded state representation as dynamics:

```python
eq = equilibrium.equilibrium(Delta0=1.0)

print(eq.Delta)
print(eq.converged)
```

The returned `GapResult` contains:

- `Delta`: self-consistent gap.
- `X`: encoded state.
- `gR`: retarded Green function.
- `gK`: Keldysh Green function fixed by equilibrium occupation.
- `current`: lowest-order Moyal current `J/sigma_n`, or `None` if disabled.
- `converged`, `iterations`, and `error`.

For a static vector potential, include `VectorPotentialDrive(A_static)` in the
model self-energy list before building the equilibrium model.

Equilibrium backend selection is controlled by `solverParams.equilibrium_method`:

- `"fixed_point"`: nested fixed-point iteration over the retarded state and gap.
  This is the current default.
- `"chi_root"`: root finding over the spectral angle `chi`, with `Delta[chi]`
  imposed inside the residual. This backend first runs the fixed-point solver
  and uses that state as the root-solver initial guess.

The `"chi_root"` backend is experimental. Its residual has multiple mathematical
roots, including the normal branch with `Delta=0`, so it can converge to the
normal branch near the transition without continuation in external parameters or
an explicit branch-selection rule.

## Dynamics

Dynamics uses an implicit backward-Euler update. Initialize first, then run:

```python
dynamics.initialize(model, Delta0=1.0)
result = dynamics.run()
```

`dynamics.initialize(...)` both stores and returns the initial `StateBundle`.

```python
bundle0 = dynamics.initialize(model, Delta0=1.0)
```

Arbitrary nonequilibrium initial states can be passed as either raw encoded
states or `StateBundle` objects:

```python
result = dynamics.run(X0=bundle0)
```

The returned `DynamicsResult` contains:

- `times`
- `X`
- `Delta`
- `converged`
- `scipy_converged`
- `residual_converged`
- `iterations`
- `residual`
- `step_time`
- `current`

## Progress Feedback

Progress reporting can be enabled through `solverParams`:

```python
numerics = ud.solverParams(
    dynamics_progress=True,
    dynamics_progress_every=5,
)
```

or directly on `run`:

```python
result = dynamics.run(progress=True, progress_every=5)
```

For custom logging, pass a callback:

```python
history = []

def log_status(status):
    history.append(status)

result = dynamics.run(progress_callback=log_status)
```

The callback receives a dictionary with:

- `step`
- `steps_total`
- `time`
- `Delta`
- `step_time`
- `elapsed`
- `eta`
- `nfev`
- `scipy_converged`
- `residual_converged`
- `residual`

## Development Notes

The current implementation favors clarity and shared algebra over maximum speed.
When optimizing, prefer changes inside `UsadelAlgebra` first so equilibrium and
dynamics inherit the same algebraic improvements.

## Reference Appendix

This section lists the current parameter and result fields in the working API.

### `gridParams`

Constructor fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `cutoff` | required | Positive frequency cutoff used to construct `ws`. |
| `nw` | required | Number of frequency grid points. |
| `time_points` | `None` | One-dimensional strictly increasing time array. If `None`, uses `[0.0]`. |

Derived fields:

| Field | Meaning |
| --- | --- |
| `ws` | Frequency grid generated by `make_frequency_grid(cutoff, nw)`. |
| `nt` | Number of time points. |
| `dt` | Uniform time step if the grid is uniform, `nan` if nonuniform, or `0.0` for a single time point. |
| `dts` | Local time steps from `np.diff(time_points)`. |
| `is_uniform_time` | Boolean flag for uniform time spacing. |
| `dt_min` | Minimum local time step. |
| `dt_max` | Maximum local time step. |
| `dw` | Frequency spacing. |

### `modelParams`

| Field | Default | Meaning |
| --- | --- | --- |
| `grid` | required | `gridParams` instance. |
| `eta` | required | Broadening/relaxation parameter entering `h0`. |
| `T` | required | Model temperature. |
| `bcs_constant` | required | Pairing constant used in the gap equation. |
| `self_energies` | `None` | `None`, a single `SelfEnergy`, or a sequence of `SelfEnergy` objects. Normalized internally to a list. |
| `self_energy_params` | `None` | Optional dictionary passed to custom self-energy hooks. |

### `solverParams`

| Field | Default |
| --- | --- |
| `retarded_maxiter` | `50` |
| `retarded_tol` | `1.0e-11` |
| `retarded_mix` | `0.7` |
| `retarded_step` | `0.05` |
| `equilibrium_method` | `"fixed_point"` |
| `equilibrium_root_maxiter` | `2000` |
| `equilibrium_root_tol` | `1.0e-10` |
| `gap_step` | `0.5` |
| `gap_tol` | `1.0e-9` |
| `gap_maxiter` | `200` |
| `dynamics_method` | `"root"` |
| `dynamics_maxiter` | `50` |
| `dynamics_tol` | `1.0e-9` |
| `dynamics_xtol` | `None` |
| `dynamics_residual_tol` | `None` |
| `dynamics_mix` | `0.7` |
| `dynamics_adaptive` | `False` |
| `dynamics_dt_min` | `None` |
| `dynamics_dt_max` | `None` |
| `dynamics_shrink` | `0.5` |
| `dynamics_growth` | `1.25` |
| `dynamics_safety` | `0.9` |
| `dynamics_progress` | `False` |
| `dynamics_progress_every` | `1` |
| `compute_current` | `True` |

### Self-Energy Objects

`SelfEnergy` constructor fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `retarded` | `None` | Optional callable returning the retarded self-energy. |
| `keldysh` | `None` | Optional callable returning the Keldysh self-energy. |

`VectorPotentialDrive` constructor fields:

| Field | Default | Meaning |
| --- | --- | --- |
| `A` | required | Float or callable `A(t)`. |

### `StateBundle`

| Field | Meaning |
| --- | --- |
| `X` | Encoded state vector. |
| `Delta` | Gap computed from, or attached to, the state. |
| `gR` | Retarded Green function associated with `X`. |
| `gK` | Keldysh Green function associated with `X`. |

### `GapResult`

| Field | Meaning |
| --- | --- |
| `Delta` | Self-consistent gap. |
| `gR` | Retarded Green function. |
| `gK` | Keldysh Green function. |
| `X` | Encoded state. |
| `converged` | Boolean convergence flag. |
| `iterations` | Number of gap iterations. |
| `error` | Final reported gap/retarded error. |
| `current` | Lowest-order Moyal current `J/sigma_n`, or `None` if disabled. |

### `DynamicsResult`

| Field | Meaning |
| --- | --- |
| `times` | Time points included in the returned trace. |
| `X` | Encoded state trace. |
| `Delta` | Gap trace. |
| `converged` | Per-step residual convergence flag. |
| `scipy_converged` | Per-step SciPy solver convergence flag. |
| `residual_converged` | Per-step residual threshold flag. |
| `iterations` | Per-step function evaluation count from the nonlinear solver. |
| `residual` | Per-step final residual norm. |
| `step_time` | Per-step wall-clock time in seconds. |
| `current` | Current trace `J/sigma_n`, or `None` if disabled. |
