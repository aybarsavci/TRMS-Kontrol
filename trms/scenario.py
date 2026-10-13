"""Reference step scenarios shared by both controllers."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

AXES = ("pitch", "yaw")


@dataclass
class Scenario:
    name: str
    duration: float
    events: list            # [(axis, t, value), ...]
    x0: np.ndarray          # initial plant state

    @classmethod
    def from_config(cls, name: str, cfg: dict, x0: np.ndarray) -> "Scenario":
        events = sorted(((str(a), float(t), float(v)) for a, t, v in cfg["events"]), key=lambda e: e[1])
        for a, _, _ in events:
            if a not in AXES:
                raise ValueError(f"Unknown axis '{a}' in scenario '{name}'")
        return cls(name=name, duration=float(cfg["duration"]), events=events, x0=np.asarray(x0, float))

    def initial_ref(self) -> dict:
        return {"pitch": float(self.x0[0]), "yaw": float(self.x0[3])}

    def reference(self, t) -> np.ndarray:
        """Reference [pitch, yaw] at time(s) ``t``; returns shape (2,) or (2, len(t))."""
        t = np.asarray(t, float)
        ref = np.empty((2,) + t.shape)
        for i, axis in enumerate(AXES):
            r = np.full(t.shape, self.initial_ref()[axis])
            for a, te, v in self.events:
                if a == axis:
                    r = np.where(t >= te - 1e-9, v, r)
            ref[i] = r
        return ref

    def steps(self) -> list:
        """Every reference change as a segment: axis, start/end time, from/to value.

        A segment ends at the next reference change on EITHER axis, so a step's
        metrics are not contaminated by the cross-coupling disturbance of a later
        step on the other axis (that disturbance is reported separately).
        """
        out = []
        all_times = sorted({t for _, t, _ in self.events})
        for i, axis in enumerate(AXES):
            prev = self.initial_ref()[axis]
            evs = [(t, v) for a, t, v in self.events if a == axis]
            for t, v in evs:
                later = [tt for tt in all_times if tt > t + 1e-9]
                t_end = later[0] if later else self.duration
                if abs(v - prev) > 1e-12:
                    out.append(dict(axis=axis, axis_index=i, t_start=t, t_end=t_end,
                                    r_from=prev, r_to=v, step_size=v - prev))
                prev = v
        return sorted(out, key=lambda s: (s["t_start"], s["axis_index"]))
