import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import matplotlib
from matplotlib.colors import BoundaryNorm

f1 = h5py.File(sys.argv[1], 'r')
ds = sys.argv[2]
o_dir = int(sys.argv[3])
step = int(sys.argv[4])

if 'PML_0' in f1[ds].keys():
    # plt.style.use('ggplot')
    gs = gridspec.GridSpec(3, 3, width_ratios=[54, 94, 54], height_ratios=[24, 54, 24], wspace=0, hspace=0)
    levels = np.arange(-1, 10, 1)
    cmap = plt.get_cmap('Set1')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    plt.subplot(gs[2, :]).pcolorfast(f1[ds + "/PML_0"][5:-5, 5:-5, 0, o_dir, step], clim=[-300, 300])
    plt.subplot(gs[1, 0]).pcolorfast(f1[ds + "/PML_2"][5:-5, 5:-5, 0, o_dir, step], clim=[-300, 300])
    plt.subplot(gs[1, 1]).pcolorfast(f1[ds +"/Center"][5:-5, 5:-5, 0, o_dir, step], clim=[-300, 300])
    plt.subplot(gs[1, 2]).pcolorfast(f1[ds + "/PML_3"][5:-5, 5:-5, 0, o_dir, step], clim=[-300, 300])
    plt.subplot(gs[0, :]).pcolorfast(f1[ds + "/PML_1"][5:-5, 5:-5, 0, o_dir, step], clim=[-300, 300])

    for i, ax in enumerate(plt.gcf().axes):
        # ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        # for tl in ax.get_xticklabels() + ax.get_yticklabels():
        #     tl.set_visible(False)
        plt.setp(ax, xticks=[], yticks=[])

else:
    plt.contour(f1[ds + "/Center"][:, :, 0, o_dir, step])

plt.show()
