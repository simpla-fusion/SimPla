import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

f1 = h5py.File(sys.argv[1], 'r')
ds = sys.argv[2]
o_dir = int(sys.argv[3])
step = int(sys.argv[4])

if 'PML_0' in f1[ds].keys():
    # plt.style.use('ggplot')
    gs = gridspec.GridSpec(3, 3,
                           width_ratios=[1, 5, 1],
                           height_ratios=[1, 5, 1]
                           )

    m_levels = np.arange(-200, 100, 5)

    plt.subplot(gs[0, 1]).imshow(f1[ds + "/PML_0"][:, :, 0, o_dir, step], clim=(-200, 100))
    plt.subplot(gs[1, 0]).imshow(f1[ds + "/PML_2"][:, :, 0, o_dir, step], clim=(-200, 100))
    plt.subplot(gs[1, 1]).imshow(f1[ds + "/Center"][:, :, 0, o_dir, step], clim=(-200, 100))
    plt.subplot(gs[1, 2]).imshow(f1[ds + "/PML_3"][:, :, 0, o_dir, step], clim=(-200, 100))
    plt.subplot(gs[2, 1]).imshow(f1[ds + "/PML_1"][:, :, 0, o_dir, step], clim=(-200, 100))

    for i, ax in enumerate(plt.gcf().axes):
        # ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        # for tl in ax.get_xticklabels() + ax.get_yticklabels():
        #     tl.set_visible(False)
        plt.setp(ax, xticks=[], yticks=[])

else:
    plt.contour(f1[ds + "/Center"][:, :, 0, o_dir, step])

plt.show()
