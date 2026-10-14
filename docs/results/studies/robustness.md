# N-MPC robustness to model mismatch

The N-MPC keeps the nominal model and the tuned weights (N = 38); only the simulated plant changes. Evaluation scenario, full-state feedback, no noise. `IAE vs nominal` = total IAE divided by the nominal case.

| case | change | diverged | IAE pitch | IAE yaw | IAE vs nominal | max OS pitch [%] | unsettled pitch | unsettled yaw | u_pitch saturated |
|---|---|---|---|---|---|---|---|---|---|
| nominal | plant = model | no | 1.88 | 1.87 | 1.00 | 0.989 | 0 | 0 | 18.8% |
| beam inertia +20% | J_v ×1.2 | no | 1.89 | 1.89 | 1.01 | 6.91 | 0 | 0 | 18.3% |
| beam inertia -20% | J_v ×0.8 | no | 2.02 | 1.88 | 1.04 | 3.49 | 0 | 0 | 18.4% |
| rotor thrust -15% | S_f ×0.85 (both rotors weaker) | no | 4.63 | 2.78 | 1.98 | 11.2 | 3 | 3 | 31.6% |
| rotor thrust +15% | S_f ×1.15 (both rotors stronger) | no | 4.24 | 2.64 | 1.84 | 80.9 | 3 | 2 | 12.2% |
| friction ×2 | K_v, K_h ×2 | no | 1.81 | 1.89 | 0.99 | 0.168 | 0 | 0 | 18.2% |
| main motor 30% slower | T_mr ×1.3 | no | 2.25 | 1.94 | 1.12 | 6.64 | 0 | 0 | 28.2% |
| counterweight +5% | A ×1.05 (rest angle moves) | no | 10.9 | 4.58 | 4.12 | 208 | 3 | 3 | 13.5% |
| combined | J_v ×1.2, S_f ×0.85, T_mr ×1.3, K_v,K_h ×2 | no | 4.75 | 2.82 | 2.02 | 18 | 3 | 3 | 34.5% |
| tail coeff. as printed | F_h linear term 0.0801 → 0.8080 (paper value) | no | 4.15 | 4.34 | 2.27 | 47.8 | 3 | 3 | 52.6% |
