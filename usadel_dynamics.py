"""Object-oriented homogeneous kinetic Usadel solver.

This module reorganizes the procedural flow in ``usadel_kinetic.py`` into
inspectable objects:

* ``PauliAlgebra`` handles Pauli-vector matrix algebra.
* ``UsadelAlgebra`` maps full states into algebraic Green functions,
  Hamiltonians, gaps, and residual ingredients.
* ``EquilibriumModel`` handles self-consistent equilibrium nonlinear solves.
* ``BackwardEulerDynamics`` handles implicit finite-difference dynamics.

All Nambu matrices are stored as Pauli vectors ``[tau0, tau1, tau2, tau3]``.
The per-frequency state is always
``X_w = [Re chi_w, Im chi_w, f0_w, f3_w]``. The spectral angle is algebraically
constrained rather than independently dynamical, but it remains part of every
state vector and solution trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Callable, Optional, Sequence

import numpy as np


###################
# Constants, Types, and Simple Helpers
###################

BCS_GAP_RATIO = 1.764 ### Ratio of Delta(0)/T_c in zero eta weak-coupling limit of BCS gap equation

Array = np.ndarray
PauliFn = Callable[..., Array]
ScalarFn = Callable[[float], float]


def make_frequency_grid(wmax: float = 8.0, nw: int = 401) -> Array:
    return np.linspace(-wmax, wmax, nw)


def guess_bcs_constant(cutoff: float, Tc: float) -> float:
    return 1.0 / np.log(2.0 * np.exp(np.euler_gamma) / np.pi * cutoff / Tc)


def frequency_gradient(a: Array, ws: Array) -> Array:
    return np.gradient(a, ws, axis=0, edge_order=2 if len(ws) > 2 else 1)


###################
# Configuration and Result Data
###################

class SelfEnergy:
    """Base class for vectorized self-energy contributions, with ``h = h0 - sigma``.

    Each method returns a Pauli-vector array with shape ``(nw, 4)`` and is called
    with keyword arguments ``ws, Delta, T, t, gR, gK, self_energy_params``.
    Optional hooks may be supplied for lightweight custom self-energies.
    """

    def __init__(
        self,
        retarded: Optional[PauliFn] = None,
        keldysh: Optional[PauliFn] = None,
    ):
        self._retarded_hook = retarded
        self._keldysh_hook = keldysh

    def retarded(self, *, ws: Array, **kwargs) -> Array:
        if self._retarded_hook is None:
            return np.zeros((len(ws), 4), dtype=complex)
        return self._retarded_hook(ws=ws, **kwargs)

    def keldysh(self, *, ws: Array, **kwargs) -> Array:
        if self._keldysh_hook is None:
            return np.zeros((len(ws), 4), dtype=complex)
        return self._keldysh_hook(ws=ws, **kwargs)


@dataclass(init=False)
class VectorPotentialDrive(SelfEnergy):
    """External vector-potential pair-breaking self-energy.

    Implements ``sigma_A = -i A(t)^2 tau3 g tau3``. The supplied ``A`` is the
    already-rescaled vector potential, including physical prefactors.
    """

    A: float | ScalarFn

    def __init__(self, A: float | ScalarFn):
        super().__init__()
        self.A = A

    def value(self, t: float) -> float:
        return float(self.A(t) if callable(self.A) else self.A)

    def gamma(self, t: float) -> float:
        A_t = self.value(t)
        return float(A_t * A_t)

    def _sigma(self, g: Optional[Array], t: float, ws: Array) -> Array:
        if g is None:
            return np.zeros((len(ws), 4), dtype=complex)
        return -1.0j * self.gamma(t) * PauliAlgebra.tau3_sandwich(g)

    def retarded(
        self,
        *,
        ws: Array,
        gR: Optional[Array],
        t: float = 0.0,
        **kwargs,
    ) -> Array:
        return self._sigma(gR, t, ws)

    def keldysh(
        self,
        *,
        ws: Array,
        gK: Optional[Array],
        t: float = 0.0,
        **kwargs,
    ) -> Array:
        return self._sigma(gK, t, ws)


@dataclass
class GapResult:
    Delta: float
    gR: Array
    gK: Array
    X: Array
    converged: bool
    iterations: int
    error: float


@dataclass
class DynamicsResult:
    times: Array
    X: Array
    Delta: Array
    converged: Array
    scipy_converged: Array
    residual_converged: Array
    iterations: Array
    residual: Array
    step_time: Array


@dataclass
class gridParams:
    """Frequency and time grids shared by initialization and dynamics models."""

    cutoff: float
    nw: int
    time_points: Optional[Array] = None
    ws: Array = field(init=False)
    nt: int = field(init=False)
    dt: float = field(init=False)
    dts: Array = field(init=False)
    is_uniform_time: bool = field(init=False)
    dt_min: float = field(init=False)
    dt_max: float = field(init=False)
    dw: float = field(init=False)

    def __post_init__(self) -> None:
        self.cutoff = float(self.cutoff)
        self.nw = int(self.nw)
        if self.nw < 2:
            raise ValueError("nw must be at least 2.")
        self.ws = make_frequency_grid(self.cutoff, self.nw)
        self.dw = float(self.ws[1] - self.ws[0])
        if self.time_points is None:
            self.time_points = np.array([0.0], dtype=float)
        else:
            self.time_points = np.asarray(self.time_points, dtype=float)
        if self.time_points.ndim != 1:
            raise ValueError("time_points must be a one-dimensional array.")
        if len(self.time_points) == 0:
            raise ValueError("time_points must contain at least one point.")
        self.nt = int(len(self.time_points))
        if self.nt == 1:
            self.dt = 0.0
            self.dts = np.array([], dtype=float)
            self.is_uniform_time = True
            self.dt_min = 0.0
            self.dt_max = 0.0
        else:
            steps = np.diff(self.time_points)
            if np.any(steps <= 0.0):
                raise ValueError("time_points must be strictly increasing.")
            self.dts = steps.astype(float)
            self.is_uniform_time = bool(np.allclose(steps, steps[0]))
            self.dt = float(steps[0]) if self.is_uniform_time else float("nan")
            self.dt_min = float(np.min(steps))
            self.dt_max = float(np.max(steps))

    def frequency_compatible_with(self, other: "gridParams") -> bool:
        return self.cutoff == other.cutoff and self.nw == other.nw


@dataclass
class modelParams:
    """Physical and model-level parameters for the Usadel algebra."""

    grid: gridParams
    eta: float
    T: float
    bcs_constant: float
    self_energies: Optional[SelfEnergy | Sequence[SelfEnergy]] = None
    self_energy_params: Optional[dict] = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate runtime model wiring.

        This is intentionally callable outside ``__post_init__`` because
        notebooks can hold stale or mutated modelParams instances.
        """
        if self.self_energies is None:
            self.self_energies = []
        elif isinstance(self.self_energies, SelfEnergy):
            self.self_energies = [self.self_energies]
        elif isinstance(self.self_energies, Sequence):
            self.self_energies = list(self.self_energies)
        else:
            raise TypeError("self_energies must be None, a SelfEnergy, or a sequence of SelfEnergy objects.")
        for sigma in self.self_energies:
            if not isinstance(sigma, SelfEnergy):
                raise TypeError("Every entry in self_energies must be a SelfEnergy instance.")


@dataclass
class solverParams:
    """Numerical controls for retarded and gap self-consistency solvers."""

    retarded_maxiter: int = 50
    retarded_tol: float = 1.0e-11
    retarded_mix: float = 0.7
    retarded_step: float = 0.05
    gap_step: float = 0.5
    gap_tol: float = 1.0e-9
    gap_maxiter: int = 200
    dynamics_method: str = "root"
    dynamics_maxiter: int = 50
    dynamics_tol: float = 1.0e-9
    dynamics_xtol: Optional[float] = None
    dynamics_residual_tol: Optional[float] = None
    dynamics_mix: float = 0.7
    dynamics_adaptive: bool = False
    dynamics_dt_min: Optional[float] = None
    dynamics_dt_max: Optional[float] = None
    dynamics_shrink: float = 0.5
    dynamics_growth: float = 1.25
    dynamics_safety: float = 0.9
    dynamics_progress: bool = False
    dynamics_progress_every: int = 1


@dataclass
class StateBundle:
    """Cached algebraic quantities attached to a state X."""

    X: Array
    Delta: float
    gR: Array
    gK: Array


###################
# Pauli Algebra
###################

class PauliAlgebra:
    """Vectorized operations on Pauli-vector matrices."""

    @staticmethod
    def conj(a: Array) -> Array:
        """Return ``tau_3 a^dagger tau_3`` for Pauli vectors."""
        out = np.conjugate(a).copy()
        out[:, 1] *= -1.0
        out[:, 2] *= -1.0
        return out

    @staticmethod
    def tau3_sandwich(a: Array) -> Array:
        """Return ``tau_3 a tau_3`` for Pauli vectors."""
        out = a.copy()
        out[:, 1] *= -1.0
        out[:, 2] *= -1.0
        return out

    @staticmethod
    def advanced(a_r: Array, is_green: bool) -> Array:
        a = PauliAlgebra.conj(a_r)
        return -a if is_green else a

    @staticmethod
    def mul(a: Array, b: Array) -> Array:
        """Vectorized product of Pauli-vector matrices."""
        out = np.empty_like(a, dtype=complex)
        av = a[:, 1:4]
        bv = b[:, 1:4]
        out[:, 0] = a[:, 0] * b[:, 0] + np.sum(av * bv, axis=1)
        out[:, 1:4] = (
            a[:, 0, None] * bv
            + b[:, 0, None] * av
            + 1.0j * np.cross(av, bv)
        )
        return out

    @staticmethod
    def comm(a: Array, b: Array) -> Array:
        return PauliAlgebra.mul(a, b) - PauliAlgebra.mul(b, a)

    @staticmethod
    def anticomm(a: Array, b: Array) -> Array:
        return PauliAlgebra.mul(a, b) + PauliAlgebra.mul(b, a)

    @staticmethod
    def advanced_component(a_r: Array, component: int, is_green: bool) -> Array:
        """Return one Pauli component of the advanced counterpart."""
        out = np.conjugate(a_r[:, component])
        if component in (1, 2):
            out = -out
        return -out if is_green else out

    @staticmethod
    def mul_component(a: Array, b: Array, component: int) -> Array:
        """Return one Pauli component of the product a b."""
        if component == 0:
            return (
                a[:, 0] * b[:, 0]
                + a[:, 1] * b[:, 1]
                + a[:, 2] * b[:, 2]
                + a[:, 3] * b[:, 3]
            )
        if component == 1:
            return (
                a[:, 0] * b[:, 1]
                + b[:, 0] * a[:, 1]
                + 1.0j * (a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2])
            )
        if component == 2:
            return (
                a[:, 0] * b[:, 2]
                + b[:, 0] * a[:, 2]
                + 1.0j * (a[:, 3] * b[:, 1] - a[:, 1] * b[:, 3])
            )
        if component == 3:
            return (
                a[:, 0] * b[:, 3]
                + b[:, 0] * a[:, 3]
                + 1.0j * (a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1])
            )
        raise ValueError("Pauli component must be 0, 1, 2, or 3.")

    @staticmethod
    def mul_component_right_advanced(
        a: Array,
        b_r: Array,
        component: int,
        b_is_green: bool,
    ) -> Array:
        """Return one component of a bA without allocating bA."""
        b0 = PauliAlgebra.advanced_component(b_r, 0, b_is_green)
        b1 = PauliAlgebra.advanced_component(b_r, 1, b_is_green)
        b2 = PauliAlgebra.advanced_component(b_r, 2, b_is_green)
        b3 = PauliAlgebra.advanced_component(b_r, 3, b_is_green)
        if component == 0:
            return a[:, 0] * b0 + a[:, 1] * b1 + a[:, 2] * b2 + a[:, 3] * b3
        if component == 1:
            return a[:, 0] * b1 + b0 * a[:, 1] + 1.0j * (a[:, 2] * b3 - a[:, 3] * b2)
        if component == 2:
            return a[:, 0] * b2 + b0 * a[:, 2] + 1.0j * (a[:, 3] * b1 - a[:, 1] * b3)
        if component == 3:
            return a[:, 0] * b3 + b0 * a[:, 3] + 1.0j * (a[:, 1] * b2 - a[:, 2] * b1)
        raise ValueError("Pauli component must be 0, 1, 2, or 3.")

    @staticmethod
    def mul_component_left_advanced(
        a_r: Array,
        b: Array,
        component: int,
        a_is_green: bool,
    ) -> Array:
        """Return one component of aA b without allocating aA."""
        a0 = PauliAlgebra.advanced_component(a_r, 0, a_is_green)
        a1 = PauliAlgebra.advanced_component(a_r, 1, a_is_green)
        a2 = PauliAlgebra.advanced_component(a_r, 2, a_is_green)
        a3 = PauliAlgebra.advanced_component(a_r, 3, a_is_green)
        if component == 0:
            return a0 * b[:, 0] + a1 * b[:, 1] + a2 * b[:, 2] + a3 * b[:, 3]
        if component == 1:
            return a0 * b[:, 1] + b[:, 0] * a1 + 1.0j * (a2 * b[:, 3] - a3 * b[:, 2])
        if component == 2:
            return a0 * b[:, 2] + b[:, 0] * a2 + 1.0j * (a3 * b[:, 1] - a1 * b[:, 3])
        if component == 3:
            return a0 * b[:, 3] + b[:, 0] * a3 + 1.0j * (a1 * b[:, 2] - a2 * b[:, 1])
        raise ValueError("Pauli component must be 0, 1, 2, or 3.")


###################
# Usadel Algebra
###################

# Future observable note:
# The leading Moyal-order current should likely live in this algebra layer,
# because it is a local-in-frequency checked-product expression built from
# gR/gK and tau3 commutators. A later implementation can expose the same kernel
# to equilibrium postprocessing and passive recording during dynamics. Keep the
# lowest-order A(t) contribution separate from future dA/dt corrections.

class UsadelAlgebra:
    """State encoding, Green-function algebra, Hamiltonians, and residuals."""

    def __init__(
        self,
        params: modelParams,
        solver_params: solverParams | None = None,
    ):
        self.params = params
        self.params.validate()
        self.solver_params = solver_params or solverParams()

    @property
    def ws(self) -> Array:
        return self.params.grid.ws

    @property
    def nw(self) -> int:
        return self.params.grid.nw

    @staticmethod
    def fd(ws: Array, T: float) -> Array:
        """Equilibrium tanh(w/2T), with a T=0 sign-function limit."""
        if T < 1.0e-8:
            return np.sign(ws)
        return np.tanh(0.5 * ws / T)

    def h0(self, Delta: float | Array) -> tuple[Array, Array]:
        Delta_arr = np.asarray(Delta, dtype=complex)
        if Delta_arr.ndim == 0:
            Delta_arr = np.full_like(self.ws, Delta_arr, dtype=complex)
        hr = np.zeros((self.nw, 4), dtype=complex)
        hk = np.zeros_like(hr)
        hr[:, 2] = -1.0j * Delta_arr
        hr[:, 3] = self.ws + 1.0j * self.params.eta
        hk[:, 3] = 2.0j * self.params.eta * self.fd(self.ws, self.params.T)
        return hr, hk

    def sigma(
        self,
        part: str,
        Delta: float,
        t: float = 0.0,
        gR: Optional[Array] = None,
        gK: Optional[Array] = None,
    ) -> Array:
        total = np.zeros((self.nw, 4), dtype=complex)
        for model in self.params.self_energies:
            fn = getattr(model, part)
            if fn is None:
                continue
            total += fn(
                ws=self.ws,
                Delta=Delta,
                T=self.params.T,
                t=t,
                gR=gR,
                gK=gK,
                self_energy_params=self.params.self_energy_params,
            )
        return total

    def sigma_terms(
        self,
        part: str,
        Delta: float,
        t: float = 0.0,
        gR: Optional[Array] = None,
        gK: Optional[Array] = None,
    ) -> list[Array]:
        """Return each self-energy contribution before summing.

        This is mostly useful for notebook diagnostics of time-dependent
        inherited self-energies.
        """
        terms = []
        for model in self.params.self_energies:
            fn = getattr(model, part)
            terms.append(
                fn(
                    ws=self.ws,
                    Delta=Delta,
                    T=self.params.T,
                    t=t,
                    gR=gR,
                    gK=gK,
                    self_energy_params=self.params.self_energy_params,
                )
            )
        return terms

    @staticmethod
    def _chi_to_gr(chi: Array) -> Array:
        """Construct ``gR = sin(chi) tau_2 + cos(chi) tau_3``."""
        gr = np.zeros((len(chi), 4), dtype=complex)
        gr[:, 2] = np.sin(chi)
        gr[:, 3] = np.cos(chi)
        return gr

    @staticmethod
    def _gr_to_chi(gR: Array) -> Array:
        """Project a retarded Green function onto the spectral angle chi."""
        return -1.0j * np.log(gR[:, 3] + 1.0j * gR[:, 2])

    def _build_gk(self, gR: Array, f: Array) -> Array:
        """Construct gK = gR F - F gA for F = f0 tau0 + f3 tau3.

        This closed form is equivalent to the generic Pauli products, but avoids
        building F/gA and two full matrix multiplications.
        """
        s = gR[:, 2]
        c = gR[:, 3]
        f0 = np.real(f)
        f3 = np.imag(f)
        out = np.zeros_like(gR, dtype=complex)
        out[:, 0] = f3 * (c + np.conjugate(c))
        out[:, 1] = 1.0j * f3 * (s + np.conjugate(s))
        out[:, 2] = f0 * (s - np.conjugate(s))
        out[:, 3] = f0 * (c + np.conjugate(c))
        return out

    def hamiltonian(
        self,
        Delta: float,
        t: float = 0.0,
        gR: Optional[Array] = None,
        gK: Optional[Array] = None,
    ) -> tuple[Array, Array]:
        hr0, hk0 = self.h0(Delta)
        sr = self.sigma("retarded", Delta, t=t, gR=gR, gK=gK)
        sk = self.sigma("keldysh", Delta, t=t, gR=gR, gK=gK)
        hr = hr0 - sr
        hk = hk0 - sk
        return hr, hk

    ###################
    # State Encoding
    ###################

    @staticmethod
    def pack_components(chi: Array, f: Array) -> Array:
        """Real state ``[Re chi, Im chi, f0, f3]`` by frequency."""
        return np.stack((np.real(chi), np.imag(chi), np.real(f), np.imag(f)), axis=1)

    @staticmethod
    def unpack_components(y: Array) -> tuple[Array, Array]:
        chi = y[:, 0] + 1.0j * y[:, 1]
        f = y[:, 2] + 1.0j * y[:, 3]
        return chi, f

    def make_state(self, chi: Array, f: Array) -> Array:
        return self.pack_components(chi, f).reshape(-1)

    def components(self, X: Array) -> tuple[Array, Array]:
        if X.size != 4 * self.nw:
            raise ValueError(f"Expected full chi/f state with size {4 * self.nw}.")
        return self.unpack_components(X.reshape((self.nw, 4)))

    def g_from_state(self, X: Array) -> tuple[Array, Array]:
        """Return ``gR, gK`` from the full encoded state X."""
        chi, f = self.components(X)
        gR = self._chi_to_gr(chi)
        gK = self._build_gk(gR, f)
        return gR, gK

    def gap_from_state(self, X: Array) -> float:
        """Return the BCS gap implied directly by ``gK(X)``."""
        _, gK = self.g_from_state(X)
        return self.gap_from_gK(gK)

    def gap_from_gK(self, gK: Array) -> float:
        gap = -0.25 * self.params.bcs_constant * np.trapz(np.imag(gK[:, 2]), self.ws)
        return float(np.real(gap))

    def h_from_state(
        self,
        X: Array,
        t: float = 0.0,
        Delta: Optional[float] = None,
    ) -> tuple[Array, Array]:
        """Return ``hR, hK`` using ``Delta(X)`` unless Delta is supplied."""
        gR, gK = self.g_from_state(X)
        Delta_eff = self.gap_from_state(X) if Delta is None else float(Delta)
        return self.hamiltonian(Delta_eff, t, gR, gK)

    ###################
    # Backward Euler Algebra: Checked Products
    ###################

    @staticmethod
    def checked_comm(
        aR: Array,
        aK: Array,
        bR: Array,
        bK: Array,
        a_is_green: bool,
        b_is_green: bool,
    ) -> tuple[Array, Array]:
        """Return checked-space commutator components."""
        aA = PauliAlgebra.advanced(aR, is_green=a_is_green)
        bA = PauliAlgebra.advanced(bR, is_green=b_is_green)
        r = PauliAlgebra.comm(aR, bR)
        k = PauliAlgebra.mul(aR, bK) + PauliAlgebra.mul(aK, bA)
        k -= PauliAlgebra.mul(bR, aK) + PauliAlgebra.mul(bK, aA)
        return r, k

    @staticmethod
    def checked_anticomm(
        aR: Array,
        aK: Array,
        bR: Array,
        bK: Array,
        a_is_green: bool,
        b_is_green: bool,
    ) -> tuple[Array, Array]:
        """Return checked-space anticommutator components."""
        aA = PauliAlgebra.advanced(aR, is_green=a_is_green)
        bA = PauliAlgebra.advanced(bR, is_green=b_is_green)
        r = PauliAlgebra.anticomm(aR, bR)
        k = PauliAlgebra.mul(aR, bK) + PauliAlgebra.mul(aK, bA)
        k += PauliAlgebra.mul(bR, aK) + PauliAlgebra.mul(bK, aA)
        return r, k

    @staticmethod
    def checked_comm_k_component(
        aR: Array,
        aK: Array,
        bR: Array,
        bK: Array,
        a_is_green: bool,
        b_is_green: bool,
        component: int,
    ) -> Array:
        """Return one Keldysh component of a checked commutator."""
        return (
            PauliAlgebra.mul_component(aR, bK, component)
            + PauliAlgebra.mul_component_right_advanced(aK, bR, component, b_is_green)
            - PauliAlgebra.mul_component(bR, aK, component)
            - PauliAlgebra.mul_component_right_advanced(bK, aR, component, a_is_green)
        )

    @staticmethod
    def checked_anticomm_k_component(
        aR: Array,
        aK: Array,
        bR: Array,
        bK: Array,
        a_is_green: bool,
        b_is_green: bool,
        component: int,
    ) -> Array:
        """Return one Keldysh component of a checked anticommutator."""
        return (
            PauliAlgebra.mul_component(aR, bK, component)
            + PauliAlgebra.mul_component_right_advanced(aK, bR, component, b_is_green)
            + PauliAlgebra.mul_component(bR, aK, component)
            + PauliAlgebra.mul_component_right_advanced(bK, aR, component, a_is_green)
        )

    def frequency_derivative_checked(
        self,
        aR: Array,
        aK: Array,
    ) -> tuple[Array, Array]:
        """Return frequency derivatives of checked retarded/Keldysh components."""
        return frequency_gradient(aR, self.ws), frequency_gradient(aK, self.ws)

    ###################
    # Retarded Spectral Constraint Helpers
    ###################

    def hR_from_gR(
        self,
        gR: Optional[Array],
        Delta: float,
        t: float = 0.0,
    ) -> Array:
        """Return hR from an explicit retarded Green function candidate."""
        hR, _ = self.hamiltonian(Delta, t, gR, None)
        return hR

    def state_from_retarded(self, X: Array, gR: Array) -> StateBundle:
        """Replace the spectral part of X by gR while preserving occupation."""
        _, f = self.components(X)
        X_out = self.make_state(self._gr_to_chi(gR), f)
        gK = self._build_gk(gR, f)
        return self.bundle_state(X_out, self.gap_from_gK(gK))

    def occupation_modes(self, X: Array) -> tuple[Array, Array]:
        """Return the Hermitian distribution eigenmodes f+ and f- from X."""
        _, f = self.components(X)
        f0 = np.real(f)
        f3 = np.imag(f)
        return f0 + f3, f0 - f3

    def bundle_state(
        self,
        X: Array,
        Delta: Optional[float] = None,
    ) -> StateBundle:
        """Return X together with cached gR, gK, and Delta(X)."""
        gR, gK = self.g_from_state(X)
        Delta_eff = self.gap_from_gK(gK) if Delta is None else float(Delta)
        return StateBundle(
            X=X,
            Delta=Delta_eff,
            gR=gR,
            gK=gK,
        )

    def spectral_residual(
        self,
        X: Array | StateBundle,
        t: float = 0.0,
        Delta: Optional[float] = None,
    ) -> Array:
        """Return the retarded algebraic residual [hR(X), gR(X)]."""
        bundle = X if isinstance(X, StateBundle) else self.bundle_state(X, Delta)
        hR, _ = self.hamiltonian(
            bundle.Delta if Delta is None else float(Delta),
            t,
            bundle.gR,
            bundle.gK,
        )
        return PauliAlgebra.comm(hR, bundle.gR)

    ###################
    # Backward Euler Algebra: Finite-Difference Residuals
    ###################

    def backward_euler_keldysh_residual(
        self,
        X_new: Array | StateBundle,
        X_old: Array | StateBundle,
        t_new: float,
        t_old: float,
        dt: float,
    ) -> Array:
        """Return the Keldysh component of the finite-difference Usadel residual."""
        new_bundle = X_new if isinstance(X_new, StateBundle) else self.bundle_state(X_new)
        old_bundle = X_old if isinstance(X_old, StateBundle) else self.bundle_state(X_old)
        hR_new, hK_new = self.h_from_state(new_bundle.X, t_new, new_bundle.Delta)
        hR_old, hK_old = self.h_from_state(old_bundle.X, t_old, old_bundle.Delta)

        dhwR, dhwK = self.frequency_derivative_checked(hR_new, hK_new)
        dgwR, dgwK = self.frequency_derivative_checked(new_bundle.gR, new_bundle.gK)
        dgR = new_bundle.gR - old_bundle.gR
        dgK = new_bundle.gK - old_bundle.gK
        dhR = hR_new - hR_old
        dhK = hK_new - hK_old

        _, term1K = self.checked_anticomm(
            dhwR,
            dhwK,
            dgR,
            dgK,
            a_is_green=False,
            b_is_green=True,
        )
        _, term2K = self.checked_anticomm(
            dgwR,
            dgwK,
            dhR,
            dhK,
            a_is_green=True,
            b_is_green=False,
        )
        _, term3K = self.checked_comm(
            hR_new,
            hK_new,
            new_bundle.gR,
            new_bundle.gK,
            a_is_green=False,
            b_is_green=True,
        )
        return 0.5 * term1K - 0.5 * term2K - 1.0j * dt * term3K

    def backward_euler_residual_vector(
        self,
        X_new: Array | StateBundle,
        X_old: Array | StateBundle,
        t_new: float,
        t_old: float,
        dt: float,
    ) -> Array:
        """Return the packed residual using explicit component formulas.

        This assumes the state parameterization gR = sin(chi) tau2 + cos(chi)
        tau3 and the corresponding closed-form gK. It computes only the four
        scalar channels used by the root solver.
        """
        new_bundle = X_new if isinstance(X_new, StateBundle) else self.bundle_state(X_new)
        old_bundle = X_old if isinstance(X_old, StateBundle) else self.bundle_state(X_old)
        hR_new, hK_new = self.h_from_state(new_bundle.X, t_new, new_bundle.Delta)
        hR_old, hK_old = self.h_from_state(old_bundle.X, t_old, old_bundle.Delta)

        dhwR, dhwK = self.frequency_derivative_checked(hR_new, hK_new)
        dgwR, dgwK = self.frequency_derivative_checked(new_bundle.gR, new_bundle.gK)
        dgR = new_bundle.gR - old_bundle.gR
        dgK = new_bundle.gK - old_bundle.gK
        dhR = hR_new - hR_old
        dhK = hK_new - hK_old

        # For structured gR with only tau2/tau3 components:
        # [hR, gR]_1 = 2 i (h2 g3 - h3 g2).
        retarded_tau1 = 2.0j * (
            hR_new[:, 2] * new_bundle.gR[:, 3]
            - hR_new[:, 3] * new_bundle.gR[:, 2]
        )

        keldysh = []
        for component in (0, 3):
            term1 = self.checked_anticomm_k_component(
                dhwR,
                dhwK,
                dgR,
                dgK,
                a_is_green=False,
                b_is_green=True,
                component=component,
            )
            term2 = self.checked_anticomm_k_component(
                dgwR,
                dgwK,
                dhR,
                dhK,
                a_is_green=True,
                b_is_green=False,
                component=component,
            )
            term3 = self.checked_comm_k_component(
                hR_new,
                hK_new,
                new_bundle.gR,
                new_bundle.gK,
                a_is_green=False,
                b_is_green=True,
                component=component,
            )
            keldysh.append(0.5 * term1 - 0.5 * term2 - 1.0j * dt * term3)

        return np.concatenate((
            np.real(retarded_tau1),
            np.imag(retarded_tau1),
            np.real(keldysh[0]),
            np.real(keldysh[1]),
        ))

    def backward_euler_residual(
        self,
        X_new: Array | StateBundle,
        X_old: Array | StateBundle,
        t_new: float,
        t_old: float,
        dt: float,
    ) -> tuple[Array, Array]:
        """Return retarded and Keldysh backward-Euler residuals."""
        new_bundle = X_new if isinstance(X_new, StateBundle) else self.bundle_state(X_new)
        retarded = self.spectral_residual(new_bundle, t_new)
        keldysh = self.backward_euler_keldysh_residual(
            new_bundle,
            X_old,
            t_new,
            t_old,
            dt,
        )
        return retarded, keldysh

    def spectral_residual_max(
        self,
        X: Array | StateBundle,
        t: float = 0.0,
        Delta: Optional[float] = None,
    ) -> float:
        return float(np.max(np.abs(self.spectral_residual(X, t, Delta))))

    def gap_residual(
        self,
        X: Array | StateBundle,
        Delta: Optional[float] = None,
    ) -> float:
        """Return Delta - Delta[gK(X)]. Zero if Delta is chosen as gap(X)."""
        bundle = X if isinstance(X, StateBundle) else self.bundle_state(X)
        Delta_eff = bundle.Delta if Delta is None else float(Delta)
        return float(Delta_eff - self.gap_from_gK(bundle.gK))

    def equilibrium_occupation_state(self) -> Array:
        """Return X with equilibrium occupation and zero initial chi."""
        f = self.fd(self.ws, self.params.T).astype(complex)
        chi = np.zeros_like(self.ws, dtype=complex)
        return self.make_state(chi, f)


###################
# Equilibrium Solvers
###################

class EquilibriumModel:
    """Equilibrium nonlinear solvers using the shared state algebra model."""

    def __init__(
        self,
        algebra: UsadelAlgebra,
        solver_params: solverParams | None = None,
    ):
        self.algebra = algebra
        self.solver_params = solver_params or algebra.solver_params

    @property
    def ws(self) -> Array:
        return self.algebra.ws

    @staticmethod
    def _normalize_retarded(hR: Array) -> Array:
        """Solve gR parallel hR with gR^2 = 1, ignoring scalar h0."""
        hv = hR[:, 1:4]
        z = np.sqrt(-np.sum(hv * hv, axis=1))
        gR = np.zeros_like(hR, dtype=complex)
        gR[:, 1:4] = -1.0j * hv / z[:, None]
        return gR

    def solve_retarded_state(
        self,
        X: Array,
        Delta: float,
        solver_params: solverParams | None = None,
    ) -> tuple[StateBundle, bool, int, float]:
        """Solve [gR, hR] = 0 and return a bundled constrained state."""
        opts = solver_params or self.solver_params
        gR = self._normalize_retarded(self.algebra.hR_from_gR(None, Delta))
        err = np.inf
        for it in range(1, opts.retarded_maxiter + 1):
            gR_new = self._normalize_retarded(
                self.algebra.hR_from_gR(gR, Delta)
            )
            err = float(np.max(np.abs(gR_new - gR)))
            gR = opts.retarded_mix * gR_new + (1.0 - opts.retarded_mix) * gR
            if err < opts.retarded_tol:
                return self.algebra.state_from_retarded(X, gR_new), True, it, err
        return self.algebra.state_from_retarded(X, gR), False, opts.retarded_maxiter, err

    def gap_from_state_at_delta(
        self,
        X: Array,
        Delta: float,
        solver_params: solverParams | None = None,
    ) -> tuple[float, StateBundle, bool, int, float]:
        bundle, converged, iterations, error = self.solve_retarded_state(
            X,
            Delta,
            solver_params,
        )
        return self.algebra.gap_from_gK(bundle.gK), bundle, converged, iterations, error

    def solve_gap(
        self,
        X: Array,
        Delta0: float = 1.0,
        solver_params: solverParams | None = None,
    ) -> GapResult:
        opts = solver_params or self.solver_params
        Delta = float(Delta0)
        err = np.inf
        target, bundle, ret_converged, _, ret_error = self.gap_from_state_at_delta(
            X,
            Delta,
            opts,
        )
        for it in range(1, opts.gap_maxiter + 1):
            next_delta = float(Delta + opts.gap_step * (target - Delta))
            target, bundle, ret_converged, _, ret_error = self.gap_from_state_at_delta(
                X,
                next_delta,
                opts,
            )
            err = abs(next_delta - Delta)
            Delta = next_delta
            if err < opts.gap_tol:
                return GapResult(
                    Delta, bundle.gR, bundle.gK, bundle.X, ret_converged, it, max(err, ret_error),
                )
        return GapResult(
            Delta, bundle.gR, bundle.gK, bundle.X, False, opts.gap_maxiter, max(err, ret_error),
        )

    def equilibrium(
        self,
        Delta0: float = 1.0,
        solver_params: solverParams | None = None,
    ) -> GapResult:
        X = self.algebra.equilibrium_occupation_state()
        return self.solve_gap(X, Delta0, solver_params)


###################
# Backward Euler Dynamics
###################

# Future solver note:
# The current implementation uses scipy.optimize.root with finite-difference
# Jacobians. Natural next optimization points are:
# 1. Add an analytic or semi-analytic Jacobian for backward_euler_residual_vector.
# 2. Split the Jacobian into sparse local-in-frequency blocks plus low-rank gap
#    coupling, especially while Delta is an algebraic functional of gK.
# 3. Consider promoting Delta to an explicit unknown to make the frequency-local
#    Jacobian cleaner, with one additional algebraic gap equation.
# 4. Exploit cases where the Keldysh equation is linear in f for fixed Delta and
#    gR, solving occupations with a linear sparse backend inside each nonlinear
#    retarded/gap iteration.
# 5. Add optional JAX/autodiff or GPU-backed residual/Jacobian evaluation behind
#    the same StateBundle/UsadelAlgebra interface, keeping NumPy as the default.
# These ideas belong either in UsadelAlgebra, for residual/Jacobian construction,
# or in BackwardEulerDynamics.step, for nonlinear-solver orchestration.

class BackwardEulerDynamics:
    """Implicit finite-difference dynamics for the encoded Usadel state X."""

    def __init__(
        self,
        algebra: UsadelAlgebra,
        solver_params: solverParams | None = None,
    ):
        self.algebra = algebra
        self.solver_params = solver_params or algebra.solver_params
        self.initial_state: Optional[StateBundle] = None

    @property
    def time_points(self) -> Array:
        return self.algebra.params.grid.time_points

    ###################
    # Solver Pipeline: Initialization
    ###################

    def initialize(
        self,
        initial: Array | StateBundle | modelParams | None = None,
        Delta0: float = 1.0,
        solver_params: solverParams | None = None,
    ) -> StateBundle:
        """Store the initial state used by run().

        Pass an encoded state X or StateBundle for arbitrary nonequilibrium
        initialization. Pass modelParams to initialize from that model's
        equilibrium state before evolving with this dynamics model.
        """
        if initial is None:
            bundle = self.algebra.bundle_state(
                self.algebra.equilibrium_occupation_state()
            )
        elif isinstance(initial, StateBundle):
            bundle = initial
        elif isinstance(initial, modelParams):
            current_params = self.algebra.params
            if not initial.grid.frequency_compatible_with(current_params.grid):
                raise ValueError("Initial and dynamics models must share compatible frequency grids.")
            params = solver_params or self.solver_params
            initial_algebra = UsadelAlgebra(initial, params)
            initial_equilibrium = EquilibriumModel(initial_algebra, params)
            equilibrium = initial_equilibrium.equilibrium(Delta0=Delta0)
            bundle = self.algebra.bundle_state(equilibrium.X)
        else:
            bundle = self.algebra.bundle_state(initial)
        self.initial_state = bundle
        return bundle

    ###################
    # Solver Pipeline: Residual Packing
    ###################

    def _residual_vector(
        self,
        X_new: Array,
        X_old: Array | StateBundle,
        t_new: float,
        t_old: float,
        dt: float,
    ) -> Array:
        return self.algebra.backward_euler_residual_vector(
            X_new,
            X_old,
            t_new,
            t_old,
            dt,
        )

    ###################
    # Solver Pipeline: Single Backward-Euler Step
    ###################

    def step(
        self,
        X_old: Array | StateBundle,
        t_old: float,
        t_new: float,
        solver_params: solverParams | None = None,
    ) -> tuple[StateBundle, bool, bool, int, float]:
        """Solve the implicit backward-Euler update from t_old to t_new."""
        opts = solver_params or self.solver_params
        old_bundle = X_old if isinstance(X_old, StateBundle) else self.algebra.bundle_state(X_old)
        dt = float(t_new - t_old)
        residual_tol = opts.dynamics_residual_tol or opts.dynamics_tol
        xtol = opts.dynamics_xtol if opts.dynamics_xtol is not None else 0.1 * opts.dynamics_tol
        if dt <= 0.0:
            raise ValueError("Backward Euler step requires t_new > t_old.")

        if opts.dynamics_method != "root":
            raise ValueError("Only dynamics_method='root' is implemented.")
        try:
            from scipy.optimize import root
        except ImportError as exc:
            raise ImportError("BackwardEulerDynamics requires scipy.optimize.root.") from exc

        def residual(y: Array) -> Array:
            return self._residual_vector(y, old_bundle, t_new, t_old, dt)

        initial_err = float(np.max(np.abs(residual(old_bundle.X))))
        if initial_err < residual_tol:
            return old_bundle, True, True, 0, initial_err

        sol = root(
            residual,
            old_bundle.X,
            method="hybr",
            options={
                "maxfev": opts.dynamics_maxiter,
                "xtol": xtol,
            },
        )
        bundle = self.algebra.bundle_state(sol.x)
        err = float(np.max(np.abs(residual(sol.x))))
        scipy_converged = bool(sol.success)
        residual_converged = bool(err < residual_tol)
        return bundle, scipy_converged, residual_converged, int(sol.nfev), err

    ###################
    # Solver Pipeline: Time Trace Driver
    ###################

    def run(
        self,
        X0: Array | StateBundle | None = None,
        time_points: Optional[Array] = None,
        solver_params: solverParams | None = None,
        progress: Optional[bool] = None,
        progress_every: Optional[int] = None,
        progress_callback: Optional[Callable[[dict], None]] = None,
    ) -> DynamicsResult:
        """Integrate over modelParams.grid.time_points unless an explicit grid is supplied."""
        times = self.time_points if time_points is None else np.asarray(time_points, dtype=float)
        if times.ndim != 1 or len(times) == 0:
            raise ValueError("time_points must be a nonempty one-dimensional array.")

        opts = solver_params or self.solver_params
        show_progress = opts.dynamics_progress if progress is None else bool(progress)
        report_every = opts.dynamics_progress_every if progress_every is None else int(progress_every)
        if report_every < 1:
            raise ValueError("progress_every must be at least 1.")
        if X0 is None:
            if self.initial_state is None:
                raise ValueError("No initial state is set. Call initialize(...) or pass X0 to run(...).")
            initial = self.initial_state
        else:
            initial = X0 if isinstance(X0, StateBundle) else self.algebra.bundle_state(X0)
        states = np.zeros((len(times), initial.X.size), dtype=float)
        deltas = np.zeros(len(times), dtype=float)
        converged = np.ones(len(times), dtype=bool)
        scipy_converged = np.ones(len(times), dtype=bool)
        residual_converged = np.ones(len(times), dtype=bool)
        iterations = np.zeros(len(times), dtype=int)
        residuals = np.zeros(len(times), dtype=float)
        step_times = np.zeros(len(times), dtype=float)

        current = initial
        states[0] = current.X
        deltas[0] = current.Delta
        run_start = time.perf_counter()
        for i in range(1, len(times)):
            step_start = time.perf_counter()
            (
                current,
                scipy_converged[i],
                residual_converged[i],
                iterations[i],
                residuals[i],
            ) = self.step(
                current,
                float(times[i - 1]),
                float(times[i]),
                opts,
            )
            converged[i] = residual_converged[i]
            step_times[i] = time.perf_counter() - step_start
            states[i] = current.X
            deltas[i] = current.Delta
            elapsed = time.perf_counter() - run_start
            steps_done = i
            steps_total = len(times) - 1
            mean_step_time = elapsed / steps_done
            steps_remaining = steps_total - steps_done
            eta_seconds = mean_step_time * steps_remaining
            status = {
                "step": i,
                "steps_total": steps_total,
                "time": float(times[i]),
                "Delta": float(current.Delta),
                "step_time": float(step_times[i]),
                "elapsed": float(elapsed),
                "eta": float(eta_seconds),
                "nfev": int(iterations[i]),
                "scipy_converged": bool(scipy_converged[i]),
                "residual_converged": bool(residual_converged[i]),
                "residual": float(residuals[i]),
            }
            if progress_callback is not None:
                progress_callback(status)
            if show_progress and (i == 1 or i == steps_total or i % report_every == 0):
                print(
                    "step "
                    f"{i}/{steps_total} "
                    f"t={times[i]:.6g} "
                    f"Delta={current.Delta:.6g} "
                    f"nfev={iterations[i]} "
                    f"res={residuals[i]:.3e} "
                    f"step={step_times[i]:.2f}s "
                    f"elapsed={elapsed:.1f}s "
                    f"eta={eta_seconds:.1f}s "
                    f"ok={bool(residual_converged[i])}"
                )
            if not converged[i]:
                states = states[: i + 1]
                deltas = deltas[: i + 1]
                converged = converged[: i + 1]
                scipy_converged = scipy_converged[: i + 1]
                residual_converged = residual_converged[: i + 1]
                iterations = iterations[: i + 1]
                residuals = residuals[: i + 1]
                step_times = step_times[: i + 1]
                times = times[: i + 1]
                break

        return DynamicsResult(
            times=times,
            X=states,
            Delta=deltas,
            converged=converged,
            scipy_converged=scipy_converged,
            residual_converged=residual_converged,
            iterations=iterations,
            residual=residuals,
            step_time=step_times,
        )


###################
# Model Construction
###################

def build_default_models(
    model_params: modelParams,
    solver_params: solverParams | None = None,
) -> tuple[
    UsadelAlgebra,
    EquilibriumModel,
    BackwardEulerDynamics,
]:
    """Construct the standard algebra/equilibrium/dynamics stack."""
    params = solver_params or solverParams()
    algebra = UsadelAlgebra(model_params, params)
    equilibrium_model = EquilibriumModel(algebra, params)
    dynamics_model = BackwardEulerDynamics(algebra, params)
    return algebra, equilibrium_model, dynamics_model
