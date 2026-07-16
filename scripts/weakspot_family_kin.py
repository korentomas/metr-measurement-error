import sys; sys.path.insert(0,".")
import numpy as np, arviz as az
from collections import Counter
from models.data_prep import load_model_data
data = load_model_data("data/processed/runs_filtered.parquet")
post = az.from_netcdf("outputs/fit_linear_robust.nc").posterior
logL = post["log_L"].mean(("chain","draw")).values
# thin-signal tasks: estimate-only AND long
is_est = np.array([data.is_estimate[data.task_idx_obs==t].any() for t in range(data.n_tasks)])
# IRT signal per task: total attempts and #models
n_att = np.zeros(data.n_tasks); n_mod = np.zeros(data.n_tasks)
np.add.at(n_att, data.task_idx_irt, data.n_attempts)
from collections import defaultdict
mods = defaultdict(set)
for t,m in zip(data.task_idx_irt, data.model_idx_irt): mods[t].add(m)
n_mod = np.array([len(mods[t]) for t in range(data.n_tasks)])
long = logL >= np.quantile(logL, 0.8)
thin = is_est & long
fam = data.task_family
famsize = np.bincount(fam, minlength=len(data.family_names))
# for thin tasks: family size, and how many family members are NON-thin (informative)
print(f"long (top 20%) estimate-only tasks: {thin.sum()}")
print(f"{'task_len(min)':>13s} {'#models':>7s} {'#attempts':>9s} {'family':>22s} {'famsize':>7s} {'informative_kin':>15s}")
for t in np.where(thin)[0][:25]:
    f=fam[t]
    kin = np.where(fam==f)[0]
    informative_kin = int(((~thin[kin]) & (kin!=t)).sum())  # family members that are not themselves thin
    print(f"{np.exp(logL[t]):>13.0f} {n_mod[t]:>7.0f} {n_att[t]:>9.0f} {data.family_names[f]:>22s} {famsize[f]:>7d} {informative_kin:>15d}")
# summary
kin_counts=[]
for t in np.where(thin)[0]:
    f=fam[t]; kin=np.where(fam==f)[0]
    kin_counts.append(int(((~thin[kin])&(kin!=t)).sum()))
kin_counts=np.array(kin_counts)
print(f"\nthin tasks with >=1 informative family member: {(kin_counts>=1).sum()}/{thin.sum()}")
print(f"thin tasks that are singletons or all-thin families: {(kin_counts==0).sum()}/{thin.sum()}")
