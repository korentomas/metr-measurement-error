import sys; sys.path.insert(0,".")
import numpy as np, arviz as az
from models.data_prep import load_model_data
data = load_model_data("data/processed/runs_filtered.parquet")
post = az.from_netcdf("outputs/fit_linear_robust.nc").posterior
logL = post["log_L"].stack(s=("chain","draw")).values   # (task,S)
eps  = post["eps"].stack(s=("chain","draw")).values
diff = logL + eps                                        # difficulty = what the horizon rides on
is_est = np.array([data.is_estimate[data.task_idx_obs==t].any() for t in range(data.n_tasks)])
lm = logL.mean(1); long = lm >= np.quantile(lm,0.8)
grp = {"long estimate-only": is_est&long, "long baseline-timed": (~is_est)&long, "all tasks": np.ones(data.n_tasks,bool)}
print(f"{'group':22s} {'n':>4s} {'sd(log_L)':>10s} {'sd(eps)':>8s} {'sd(difficulty)':>14s} {'corr(logL,eps)':>15s}")
for name,m in grp.items():
    sdL=logL[m].std(1).mean(); sdE=eps[m].std(1).mean(); sdD=diff[m].std(1).mean()
    # avg within-task posterior corr between logL and eps
    corr=np.mean([np.corrcoef(logL[t],eps[t])[0,1] for t in np.where(m)[0]])
    print(f"{name:22s} {m.sum():>4d} {sdL:>10.3f} {sdE:>8.3f} {sdD:>14.3f} {corr:>15.2f}")
print("\nReading: if sd(difficulty) << sd(log_L) for long estimate tasks, the horizon-relevant")
print("quantity is pinned by IRT even though the length/eps split is not (they trade off,")
print("corr ~ -1). That is why annotation inflation (Barry's worry) doesn't reach the horizon.")
