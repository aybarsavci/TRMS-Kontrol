"""Physical parameters of the Feedback Instruments TRMS 33-220.

Source of the model structure and every numerical value below:
    "Fault-Tolerant Control of a 2 DOF Helicopter (TRMS System) Based on
    H_infinity", arXiv:1306.4883, Eq. (30) and Table I.
    https://arxiv.org/abs/1306.4883

Known errata handled here (all are flagged in README.md):
  * The linear coefficient of the tail-rotor thrust polynomial F_h(w) is printed
    as 0.8080 in the paper. The Feedback 33-220 manual and most derived works
    use 0.0801; 0.8080 makes the tail rotor ~4x stronger than the main rotor,
    which is physically implausible. 0.0801 is used by default.
  * Eq. (18) prints u_vv where u_hh is meant (tail motor polynomial). The
    polynomial is applied to the tail-motor state u_hh here.

All polynomial coefficients are stored highest power first (numpy.polyval order).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math


@dataclass(frozen=True)
class TRMSParams:
    # Lumped mass/geometry moments (kg*m^2), Table I
    A: float = 0.0946875
    B: float = 0.11046
    C: float = 0.01986
    D: float = 0.04988
    E: float = 0.004745
    F: float = 0.006230
    H: float = 0.048210
    # Inertias (kg*m^2)
    J_v: float = 0.055448      # beam inertia about the horizontal (pitch) axis
    J_mr: float = 1.6543e-5    # main rotor + motor inertia
    J_tr: float = 2.65e-5      # tail rotor + motor inertia
    # Geometry (m) and thrust scaling
    l_m: float = 0.24
    l_t: float = 0.25
    S_f: float = 8.43318e-4
    # DC motor first-order lags
    T_mr: float = 1.432
    T_tr: float = 0.3842
    K_mr: float = 1.0
    K_tr: float = 1.0
    # Viscous friction
    K_v: float = 0.0095
    K_h: float = 0.00545371
    g: float = 9.81
    # Static characteristics (highest power first)
    omega_m_poly: tuple = (90.90, 599.73, -129.26, -1238.64, 63.45, 1238.41, 0.0)
    F_v_poly: tuple = (-3.48e-12, 1.09e-9, 4.123e-6, -1.632e-4, 9.544e-2, 0.0)
    omega_t_poly: tuple = (2020.0, 194.69, -4283.15, -262.87, 3796.83, 0.0)
    F_h_poly: tuple = (-3.0e-14, 1.595e-11, 2.511e-7, -1.808e-4, 0.0801, 0.0)
    # Actuator (normalised control voltage) limits
    u_min: float = -1.0
    u_max: float = 1.0

    def __post_init__(self):
        """Reject parameter sets that would divide by zero or make the model ill-posed.

        Every division in ``model.terms`` is by J_v, T_mr, T_tr or J_h(alpha_v);
        J_h = D cos^2 + E sin^2 + F lies between min(D, E) + F and max(D, E) + F,
        so it stays positive for every pitch angle iff F + min(D, E) > 0.
        """
        problems = []
        for name in ("J_v", "T_mr", "T_tr", "l_m", "l_t", "S_f"):
            if not getattr(self, name) > 0:
                problems.append(f"{name} must be > 0 (got {getattr(self, name)})")
        for name in ("K_v", "K_h", "J_mr", "J_tr", "H"):
            if getattr(self, name) < 0:
                problems.append(f"{name} must be >= 0 (got {getattr(self, name)})")
        if not self.F + min(self.D, self.E) > 0:
            problems.append("F + min(D, E) must be > 0 so that J_h(alpha_v) never vanishes")
        if not self.u_min < self.u_max:
            problems.append(f"u_min must be < u_max (got {self.u_min}, {self.u_max})")
        for name in ("omega_m_poly", "F_v_poly", "omega_t_poly", "F_h_poly"):
            if len(getattr(self, name)) < 1:
                problems.append(f"{name} must have at least one coefficient")
        if problems:
            raise ValueError("Invalid TRMS parameters:\n  " + "\n  ".join(problems))

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items()}

    @property
    def rest_pitch(self) -> float:
        """Stable pitch equilibrium with rotors off (rad).

        Gravity torque (A-B)cos(a) - C sin(a) vanishes where tan(a) = (A-B)/C.
        ``atan2`` picks the stable root for any sign of C and cannot divide by
        zero (identical to atan((A-B)/C) for the C > 0 of the 33-220).
        """
        return math.atan2(self.A - self.B, self.C)


STATE_NAMES = ("pitch", "S_v", "u_vv", "yaw", "S_h", "u_hh")
STATE_DESCRIPTIONS = {
    "pitch": "alpha_v, pitch (vertical) angle [rad]",
    "S_v": "S_v, pitch momentum state [rad/s]",
    "u_vv": "u_vv, main motor internal voltage state [-]",
    "yaw": "alpha_h, yaw (horizontal) angle [rad]",
    "S_h": "S_h, yaw momentum state [rad/s]",
    "u_hh": "u_hh, tail motor internal voltage state [-]",
}
