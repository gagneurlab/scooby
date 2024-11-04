from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pyranges as pr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap


def plot_coverage(ax, interval, y, title, ylim, color_map):
    ax.fill_between(
        np.linspace(interval[0], interval[1], num=len(y[interval[0] : interval[1]])),
        y[interval[0] : interval[1]],
        color=color_map[title.split(":")[0]],
        alpha=(0.5 if "Alternative" in title else 1)
    )
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
    ax.set_ylim(0, ylim)
    ax.set_xlim(interval)


def plot_bed(ax, bed, interval, title, strand=None):
    boxes = [Rectangle((start, 2), end - start, 1) for (start, end) in bed]
    if strand:
        color = "#4D4D4D" if strand == "+" else "#4D4D4D"
    else:
        color = "#B3B3B3"

    pc = PatchCollection(boxes, facecolor=color, edgecolor=color)
    ax.add_collection(pc)
    ax.set_xlim(interval)
    ax.set_ylim((0, 5))
    ax.set_title(title)
    ax.set_axis_off()


def plot_line(ax, interval, y, title, ylim):
    ax.plot(
        np.linspace(interval[0], interval[1], num=len(y[interval[0] : interval[1]])),
        y[interval[0] : interval[1]],
        color="#68AEDA",
    )
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
    # ax.set_ylim(0,ylim)

def add_heatmap_to_axes(out, pos, values, color, title, vmax=  None):
    vmax = values.max() * 0.001 if vmax is None else vmax
    
    # Convert hex color to RGB
    rgb_color = matplotlib.colors.hex2color(color)
    
    # Create a list of colors from white to the target color
    colors = [(1, 1, 1), rgb_color]  # (1, 1, 1) is white
    
    # Create the colormap
    cmap = LinearSegmentedColormap.from_list("blabla", colors)
    fig_out, ax_out = out
    to_heatmap = ax_out[pos]
    to_heatmap.clear()
    print (np.quantile(values, 0.995), values.max())
    ax = sns.heatmap(values, ax=to_heatmap, cmap = cmap, cbar= False, vmax = vmax )
    for k,(_, spine) in enumerate(ax.spines.items()):
        if k  == 2 or k == 0:
            spine.set_visible(True)
    #ax.set_xticks([])
    to_heatmap.set_title(title)
    to_heatmap.set_xticks([])
    to_heatmap.set_yticks([100,0], labels =['Neighbor 0', "Neighbor 100"])
    to_heatmap.spines['bottom'].set_color(color)




def plot_tracks(tracks, interval, height=1.5, color_map={}, fig_title=None, save_name=None, annotation_plot=None, annotation_scale=5):
    if not annotation_plot:
        fig, axes = plt.subplots(len(tracks), 1, figsize=(15, height * len(tracks)))
        for i,ax in enumerate(axes[1:-1]):
            if  not i == 0:
                if not i == 2:
                    ax.sharex(axes[1])
    else:
        fig, axes = plt.subplots(
            len(tracks) + 1,
            1,
            figsize=(15, height * len(tracks) + annotation_scale * height),
            gridspec_kw={"height_ratios": [1] * len(tracks) + [annotation_scale]},
        )
        for i,ax in enumerate(axes[1:-1]):
            if  not i == 0:
                if not i == 2:
                    ax.sharex(axes[1])
        annotation_plot.plot(
            None,
            width=20,
            height=5,#annotation_scale * height,
            raster=True,
            ax_var=axes[-1],
            fig=fig,
            intron_scale=1.0,
            annotation_scale=5,
            stroke_scale=1.0
        )

    if len(tracks[0]) == 4:
        for ax, (title, y, track_type, ylim) in zip(axes, tracks):
            if track_type == "coverage":
                plot_coverage(ax, interval, y, title, ylim, color_map)
            elif "bed" in track_type:
                strand = None
                if "+" in track_type:
                    strand = "+"
                elif "-" in track_type:
                    strand = "-"
                plot_bed(ax, y, interval, title, strand)
            elif track_type == "line":
                plot_line(ax, interval, y, title, ylim)

    else:
        for ax, track_list in zip(axes, tracks):
            for track in track_list:
                title, y, track_type, ylim = track[0]
                if track_type == "coverage":
                    plot_coverage(ax, interval, y, title, ylim, color_map)
                elif "bed" in track_type:
                    strand = None
                    if "+" in track_type:
                        strand = "+"
                    elif "-" in track_type:
                        strand = "-"
                    plot_bed(ax, y, interval, title, strand)
                elif track_type == "line":
                    plot_line(ax, interval, y, title, ylim)
                
    fig.suptitle(fig_title, fontsize=20, y=1)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)

    return fig, axes
