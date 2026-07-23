"""Discriminate the eps-vs-length-shrinkage fork (Barry's note).

Fork: a long, human-slow, model-easy task -- is its length OVERESTIMATED
(Barry: shrink L) or is it genuinely long but EASY for its length (my model:
negative eps)? The two are only jointly identified by the IRT layer; the
measurement (timing) layer separates them, and it does so well only for
tasks with several timed runs. So the WELL-TIMED tasks carry the ground-truth
length-vs-difficulty relationship, and the question is whether the
poorly-timed long tasks obey it or violate it.

Test: on well-timed tasks (>=3 timed runs, length pinned by data), regress
eps on length. Then ask whether the poorly-timed long tasks (estimate/1-run,
top length quartile) have eps CONSISTENT with that trend, or anomalously
negative (which would mean eps is absorbing length-overestimation -> Barry).
"""
import sys; sys.path.insert(0,".")
import numpy as np, arviz as az
from collections import Counter
from models.data_prep import load_model_data

data = load_model_data("data/processed/runs_filtered.parquet")
post = az.from_netcdf("outputs/fit_linear_robust.nc").posterior
eps = post["eps"].mean(("chain","draw")).values
logL = post["log_L"].mean(("chain","draw")).values
base_mask = ~data.is_estimate & ~data.is_censored
nruns = np.array([Counter(data.task_idx_obs[base_mask].tolist()).get(t,0) for t in range(data.n_tasks)])
is_est = np.array([data.is_estimate[data.task_idx_obs==t].any() for t in range(data.n_tasks)])

well = nruns>=3                     # length pinned by data
poor = (nruns<=1)                   # estimate-only or single run
long_q = logL >= np.quantile(logL,0.75)

# 1. On well-timed tasks: is eps correlated with length? (the extrapolatable trend)
w = well
b,a = np.polyfit(logL[w], eps[w], 1)     # eps ~ a + b*length
r = np.corrcoef(logL[w], eps[w])[0,1]
print(f"WELL-TIMED tasks (n={w.sum()}): eps vs length slope b={b:+.3f}, corr={r:+.3f}")
print(f"  => on tasks whose length we KNOW, longer tasks are {'HARDER' if b>0 else 'EASIER'} for their length")
print(f"     (Barry's fork needs long tasks to be genuinely EASIER, i.e. b<0, for my reading to hold)")

# 2. Poorly-timed long tasks: eps vs what the well-timed trend predicts
pl = poor & long_q
pred = a + b*logL[pl]
excess = eps[pl] - pred                  # negative excess = more-easy than trend => absorbed length overestimate
print(f"\nPOORLY-TIMED LONG tasks (n={pl.sum()}): actual eps vs well-timed-trend prediction")
print(f"  mean eps          {eps[pl].mean():+.3f}")
print(f"  mean predicted    {pred.mean():+.3f}")
print(f"  mean EXCESS (actual-pred) {excess.mean():+.3f}  (<0 => anomalously easy => consistent with Barry)")
from scipy import stats
t,p = stats.ttest_1samp(excess,0)
print(f"  t-test excess=0: t={t:.2f}, p={p:.3f}")

# 3. Direct: eps of long tasks, well vs poorly timed
print(f"\nLONG tasks eps by timing quality:")
print(f"  well-timed long (n={(well&long_q).sum()}): mean eps {eps[well&long_q].mean():+.3f}")
print(f"  poorly-timed long (n={(poor&long_q).sum()}): mean eps {eps[poor&long_q].mean():+.3f}")
print(f"  difference (poor-well): {eps[poor&long_q].mean()-eps[well&long_q].mean():+.3f}")
print("  (poorly-timed long systematically MORE negative => eps absorbing length overestimation)")
