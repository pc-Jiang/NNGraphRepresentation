import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from configs.configs_global import FIG_DIR

os.environ['NUMEXPR_MAX_THREADS'] = '16'
plt.rcParams.update({'font.size': 8})
line_styles = ['-', '--', ':']
# Default colors
colors = ['red', 'lightblue', 'green', 'lightgreen', 'blue', 'tomato']

os.makedirs(FIG_DIR, exist_ok=True)


def heatmap_plot(exp_name, distance_dict, row_names):
    n = len(distance_dict)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1:
        axes = [axes, ]

    vmin = []
    vmax = []
    for v in distance_dict.values():
        vmin.append(np.min(v)) 
        vmax.append(np.max(v))
    
    vmin = min(vmin)
    vmax = max(vmax)

    base_color = sns.color_palette("Blues", 1)[0]
    cmap = sns.light_palette(base_color, as_cmap=True)
    for i, (k, v) in enumerate(distance_dict.items()):
        sns.heatmap(data=v,
                cmap=cmap,
                annot=True,
                fmt=".2f",
                cbar=False,
                xticklabels=row_names,
                yticklabels=row_names if i==0 else [],
                vmin=vmin,
                vmax=vmax,
                ax=axes[i])
        axes[i].set_title(k, fontsize=12)
        axes[i].set_aspect(1)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].tick_params(axis='y', rotation=45)
    plt.tight_layout()
    plt.savefig(osp.join(FIG_DIR, exp_name+'.pdf'))
    plt.close()
