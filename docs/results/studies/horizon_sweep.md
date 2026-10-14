# N-MPC prediction-horizon sweep

Evaluation scenario, tuned weights fixed, only N changes. Ts = 0.05 s, real-time budget = 50 ms per sample. Tuned horizon: N = 38.

| N | horizon [s] | tuned | diverged | IAE pitch | IAE yaw | max OS pitch [%] | unsettled pitch | unsettled yaw | solve mean [ms] | solve max [ms] | over budget |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.50 | no | yes | 20.1 | 3.85e+04 | 116 | 3 | 3 | 3.9 | 47 | 0.0% |
| 15 | 0.75 | no | yes | 14.3 | 1.38e+04 | 56.7 | 3 | 3 | 4.6 | 57 | 0.1% |
| 20 | 1.00 | no | yes | 12.4 | 2.79e+03 | 0.278 | 2 | 2 | 6.6 | 50 | 0.0% |
| 25 | 1.25 | no | no | 3.23 | 2 | 4.31 | 0 | 0 | 6.9 | 83 | 0.3% |
| 30 | 1.50 | no | no | 2.09 | 1.92 | 3.78 | 0 | 0 | 7.3 | 127 | 0.8% |
| 38 | 1.90 | yes | no | 1.88 | 1.87 | 0.989 | 0 | 0 | 7.7 | 80 | 0.6% |
| 50 | 2.50 | no | no | 1.82 | 1.85 | 1.01 | 0 | 0 | 9.2 | 166 | 0.7% |

Solve times are wall-clock and vary between runs and machines; the trajectories and metrics are deterministic.
