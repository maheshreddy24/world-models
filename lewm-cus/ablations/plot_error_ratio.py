"""Probe error relative to copy (err / err_copy = 1 - skill), log y-axis, clean vs mixed probe, one column.

    python ablations/plot_error_ratio.py      # after probe_mixed.py; writes checkpoints/skill_plots/error_ratio_compare_log.png
"""
import sys, json; from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; sys.path[:0] = [str(ROOT), str(ROOT / 'ablations')]
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from probe_mixed import NAMES
from src.tasks import get_task
runs=[ROOT / p for p in ['checkpoints/ckpts_dynamics/double_pend_lwm','checkpoints/ckpts_dynamics/pendulum-swingup/vit','checkpoints/cartpole-swingup/vit']]
panels=[]
for r in runs:
    res=json.loads((r/'probe_mixed/rollout.json').read_text()); t=get_task(res['task'])
    for q in t.quantities: panels.append((res,t,q))
fig,axes=plt.subplots(len(panels),1,figsize=(8,3.3*len(panels)),sharex=True)
for ax,(res,t,q) in zip(axes,panels):
    h=np.array(res['horizon']); copy=np.array(res[f'{q.name}_copy'])
    for c,color,mk,lab in (("tf","#2a78d6","s","teacher-forced"),("ar","#eb6834","o","autoregressive")):
        for p,ls,mfc in (("clean","-",None),("mixed","--","white")):
            ax.plot(h,np.array(res[f'{q.name}_{p}_{c}'])/copy,f"{mk}{ls}",color=color,ms=3,mfc=mfc or color,label=f"{lab}, {p} probe")
    ax.axhline(1,color="gray",lw=1.5,ls="--",label="copy: true state held")
    ax.set_yscale("log")
    ax.set(title=f"{NAMES[t.name]}: {q.name} ({q.unit})",ylabel="err / err_copy (log)")
    ax.grid(True,which="both",alpha=0.25); ax.spines[["top","right"]].set_visible(False)
axes[-1].set_xlabel("horizon (latent steps of 5 rows)")
fig.suptitle("probe error relative to copy, clean vs mixed probe (30% predicted)\nlower is better; 1 = copy, skill = 1 - err/err_copy",y=0.998)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc="upper center",ncol=3,frameon=False,fontsize=8,bbox_to_anchor=(0.5,0.975))
fig.tight_layout(rect=(0,0,1,0.955)); out=ROOT / ('checkpoints/skill_plots/error_ratio_compare_log.png'); fig.savefig(out,dpi=130); print(out)
