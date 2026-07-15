import sys; sys.path.insert(0,".")
import numpy as np, arviz as az
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from models.data_prep import load_model_data
data = load_model_data("data/processed/runs_filtered.parquet")
fam = az.from_netcdf("outputs/fit_linear_robust_fameps.nc").posterior
fam_eff = (fam["fam_raw"]*fam["sigma_eps_fam"]).stack(s=("chain","draw")).values  # (family,S)
names = data.family_names
sizes = np.bincount(data.task_family, minlength=len(names))
med = np.median(fam_eff,axis=1)
lo = np.quantile(fam_eff,0.025,axis=1); hi=np.quantile(fam_eff,0.975,axis=1)
# only families with >=2 tasks and CI excluding 0 (confident)
mask = (sizes>=2)
idx = np.where(mask)[0]
idx = idx[np.argsort(med[idx])]
fig, axes = plt.subplots(1,2,figsize=(15,6.5),gridspec_kw={'width_ratios':[2,1]})
ax=axes[0]
y=np.arange(len(idx))
colors=['C3' if med[i]>0 else 'C0' for i in idx]
ax.errorbar(med[idx],y,xerr=[med[idx]-lo[idx],hi[idx]-med[idx]],fmt='o',ms=4,
            ecolor='gray',elinewidth=1,capsize=0,color='none')
ax.scatter(med[idx],y,c=colors,s=28,zorder=3)
ax.axvline(0,color='black',lw=0.8)
ax.set_yticks(y); ax.set_yticklabels([f"{names[i]} (n={sizes[i]})" for i in idx],fontsize=7)
ax.set_xlabel("family difficulty effect on eps (log-min): + harder / - easier than length predicts")
ax.set_title("A. Task-family residual-difficulty effects (families with >=2 tasks)")
# secondary x in x-fold
for xf,lab in [(-4.6,"100x easier"),(0,"as length"),(4.6,"100x harder")]:
    pass
ax=axes[1]
sf=fam["sigma_eps_fam"].values.ravel(); sw=fam["sigma_eps_within"].values.ravel()
frac=fam["eps_between_frac"].values.ravel()
ax.hist(frac,bins=40,color='C2',alpha=0.8)
ax.axvline(np.median(frac),color='black',lw=2,label=f"median {np.median(frac):.2f}")
ax.set_xlabel("between-family share of residual-difficulty variance")
ax.set_title("B. ~67% of the '8x' is predictable\nfamily structure, not irreducible noise")
ax.legend()
ax.text(0.02,0.97,f"sigma_eps_fam    = {np.median(sf):.2f} (between)\nsigma_eps_within = {np.median(sw):.2f} (within, ~irreducible)\nflat sigma_eps   = 2.16",
        transform=ax.transAxes,va='top',fontsize=9,family='monospace',
        bbox=dict(boxstyle='round',fc='white',ec='gray'))
fig.tight_layout()
fig.savefig("outputs/figures/eps_family_decomposition.png",dpi=130)
print("wrote figure; confident families (CI excludes 0):",int(((lo>0)|(hi<0))[mask].sum()),"of",mask.sum())
