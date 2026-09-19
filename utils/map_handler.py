from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def gen_cmap(N=256):
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        '',
        [
            (0, 'black'),
            (0.06, 'blue'),
            (0.23, '#2ab6c6'),
            (0.38, 'yellow'),
            (0.6, 'red'),
            (1, 'white'),
        ],
        N=N,
    )


def plot_2dmap(
    img,
    show_colorbar=True,
    show_cup=False,
    with_ref=False,
    ref_img=None,
    delartifact=False,
    cm=None,
    title_on=True,
    title='',
):
    if cm is None:
        cm = gen_cmap(256)

    img_copy2d = deepcopy(img)
    if show_cup:
        img_copy2d[img_copy2d == -1] = np.nan
        cm.set_bad('gray')
        cm.set_under(color='lightgray')
    if with_ref:
        img_copy2d[ref_img == -2] = -2
        img_copy2d[ref_img == -1] = np.nan
        cm.set_bad(color='gray')
        cm.set_under(color='lightgray')
    if delartifact:
        img_copy2d[(img_copy2d <= 30) & (img_copy2d > 0)] = 0

    fig = plt.figure()
    ax = plt.subplot(111)
    img_copy2d = ax.imshow(img_copy2d, cmap=cm, vmin=-0.00000001, vmax=350)
    if show_colorbar:
        cbar = plt.colorbar(img_copy2d, pad=0.01, aspect=12, location='left')
        cbar.set_ticks([0, 175, 350])
        cbar.ax.set_yticklabels(['0 μm', '175 μm', '350 μm'])
        cbar.ax.tick_params(labelsize=14)
        if title_on:
            ax.set_title(title, fontsize=15)
        ax.axis('off')
    else:
        ax.axis('off')
    plt.show()
