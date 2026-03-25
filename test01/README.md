# Test 01 — Convergence Efficiency

## What
Compares ME-CFL with Hegazy compression (242 bytes/round) against Baseline FedAvg with no compression (2212 bytes/round).

## Metrics
- Accuracy vs Cumulative KB sent
- Accuracy vs Cumulative Energy (Joules)
- MAE convergence over 7 days
- Per-round cost (bytes, packets, energy)

## How to Run
```bash
cd ~/LoRa_Fl_Compression.proj
python3 test01/run_test01.py
```

Requires `hardwaretest03/` with `data/` folder.

## Output
Results saved to `test01/results/`:
- `fig1_dual_axis_accuracy_energy.pdf` — Accuracy + Energy vs Rounds
- `fig2_accuracy_vs_kb.pdf` — Accuracy vs Cumulative KB
- `fig3_mae_convergence.pdf` — MAE over days
- `fig4_cost_comparison.pdf` — Bar chart: bytes and packets per round

## Expected Result
ME-CFL reaches 95% accuracy with ~90% less data and energy than baseline.
