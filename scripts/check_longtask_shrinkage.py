import sys; sys.path.insert(0,"."); import numpy as np, arviz as az
from models.data_prep import load_model_data
data = load_model_data("data/processed/runs_filtered.parquet")
post = az.from_netcdf("outputs/fit_linear_robust.nc").posterior
logL = post["log_L"].mean(("chain","draw")).values  # posterior mean per task
# raw per-task observed length: mean log_dur of that task's obs (baseline or estimate)
raw = np.full(data.n_tasks, np.nan)
for t in range(data.n_tasks):
    m = data.task_idx_obs==t
    if m.any(): raw[t]=np.mean(data.log_dur[m])
shift = logL - raw
# focus on the longest tasks (Barry: longest are overestimates -> should shrink DOWN)
order = np.argsort(-raw)
print("Longest-annotation tasks: does the model shrink them DOWN (Barry's prediction)?")
print(f"{'rank':>4s} {'raw(min)':>10s} {'postmean(min)':>13s} {'shift(log)':>10s} {'estimate?':>9s}")
for r,t in enumerate(order[:12]):
    est = data.is_estimate[data.task_idx_obs==t].any()
    print(f"{r:>4d} {np.exp(raw[t]):>10.1f} {np.exp(logL[t]):>13.1f} {shift[t]:>+10.2f} {str(bool(est)):>9s}")
# aggregate shrinkage among the top decile longest
top = order[:23]
print(f"\ntop-decile longest tasks: mean shift {shift[top].mean():+.3f} log-min ({np.exp(shift[top].mean()):.2f}x)")
print(f"all tasks: mean |shift| {np.nanmean(np.abs(shift)):.3f}")
# is there systematic regression-to-mean? corr of shift with raw length
good=~np.isnan(shift)
print(f"corr(shift, raw length): {np.corrcoef(shift[good], raw[good])[0,1]:+.3f}  (negative = long tasks pulled down)")
