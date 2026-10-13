# TRMS reference tracking: PID vs N-MPC

Run: reference run (python main.py)  
Created: 2026-10-14T00:19:59  
Evaluation scenario: pitch -0.20 rad @ 0s, yaw +0.50 rad @ 10s, pitch -0.05 rad @ 20s, yaw -0.50 rad @ 30s, pitch -0.40 rad @ 40s, yaw +0.00 rad @ 50s

## Whole-run metrics (lower is better)

| Metric | PID | N-MPC |
|---|---|---|
| IAE pitch [rad·s] | 10.2 | 1.88 |
| IAE yaw [rad·s] | 8.29 | 1.87 |
| RMSE pitch [rad] | 0.248 | 0.0865 |
| RMSE yaw [rad] | 0.246 | 0.124 |
| Mean overshoot pitch [%] | 44.5 | 0.585 |
| Mean overshoot yaw [%] | 0.231 | 2.62 |
| Mean settling time pitch, settled steps only [s] | not settled | 1.62 |
| Mean settling time yaw, settled steps only [s] | 6.95 | 2.82 |
| Steps not settled (pitch / yaw) | 3 / 2 | 0 / 0 |
| Control effort ∫u² (total) | 25.4 | 33.7 |
| Input total variation (total) | 11 | 44.6 |
| Saturated samples pitch [%] | 0.7 | 18.8 |
| Compute time mean / max [ms] | 0.06 / 0.6 | 10.78 / 113.0 |

## Per-step results

| Step | OS% PID | Ts2% PID [s] | OS% N-MPC | Ts2% N-MPC [s] |
|---|---|---|---|---|
| pitch @0s (-38.5°→-11.5°) | 25.5 | not settled | 0.177 | 1.75 |
| yaw @10s (0.0°→28.6°) | 0 | not settled | 2.38 | 2.5 |
| pitch @20s (-11.5°→-2.9°) | 31.9 | not settled | 0.59 | 1.8 |
| yaw @30s (28.6°→-28.6°) | 0.693 | 6.95 | 5.49 | 4.1 |
| pitch @40s (-2.9°→-22.9°) | 76 | not settled | 0.989 | 1.3 |
| yaw @50s (-28.6°→0.0°) | 0 | not settled | 3.94e-10 | 1.85 |

## Controller parameters

```json
{
  "PID": {
    "type": "PID",
    "gains": {
      "Kp_pitch": 0.4560607613606723,
      "Ki_pitch": 0.39567781934057955,
      "Kd_pitch": 0.13248117829413555,
      "Kp_yaw": 0.18134776470396638,
      "Ki_yaw": 0.0010074048021299277,
      "Kd_yaw": 0.6506151121920789
    },
    "Ts": 0.05,
    "derivative_filter_tf": 0.1,
    "anti_windup": "conditional integration + integrator clamp to actuator range"
  },
  "N-MPC": {
    "type": "N-MPC",
    "solver": "CasADi/IPOPT multiple shooting",
    "horizon_steps": 38,
    "horizon_seconds": 1.9000000000000001,
    "weights": {
      "q_pitch": 1.0,
      "q_yaw": 3.048927039352034,
      "q_rate_pitch": 0.004183988622424465,
      "q_rate_yaw": 0.3429316955397997,
      "r_pitch": 0.012760717739783453,
      "r_yaw": 0.35230520706064267
    },
    "terminal_weight": 5.0,
    "prediction_integrator": "RK4 x2 substeps",
    "reference_preview": false,
    "Ts": 0.05,
    "warm_start": "shifted primal + shifted multipliers"
  }
}
```
