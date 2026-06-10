# Usadel Dynamics API

This document describes the object-oriented interface in `usadel_dynamics.py`.
It is intended to become the public API documentation for the repository as the
solver stabilizes.

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

The encoded state is always the full per-frequency vector

```text
X_w = [Re chi_w, Im chi_w, f0_w, f3_w].
```

The spectral degree of freedom `chi_w` is algebraically constrained, but it is
kept in the state so equilibrium and dynamics use the same representation.

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
- `converged`, `iterations`, and `error`.

For a static vector potential, include `VectorPotentialDrive(A_static)` in the
model self-energy list before building the equilibrium model.

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
