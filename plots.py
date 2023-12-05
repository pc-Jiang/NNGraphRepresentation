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


def heatmap_plot(exp_name, distance_dict, row_names, normalize=False):

    fig_size = list(distance_dict.values())[0].shape[0]

    base_color = sns.color_palette("Blues", 1)[0]
    cmap = sns.light_palette(base_color, as_cmap=True)
    for k, v in distance_dict.items():
        fig, axes = plt.subplots(1, 1, figsize=(fig_size, fig_size))
        if normalize:
            v = v / np.max
        vmin = np.min(v)
        vmax = np.max(v)
        sns.heatmap(data=v,
                cmap=cmap,
                annot=True,
                fmt=".2f",
                cbar=False,
                xticklabels=row_names,
                yticklabels=row_names,
                vmin=vmin,
                vmax=vmax,
                ax=axes)
        axes.set_title(k, fontsize=12)
        axes.set_aspect(1)
        axes.tick_params(axis='x', rotation=45)
        axes.tick_params(axis='y', rotation=45)
        plt.tight_layout()
        plt.savefig(osp.join(FIG_DIR, exp_name+k+'.pdf'))
        plt.close()
