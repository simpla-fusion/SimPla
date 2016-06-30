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
    plt.style.use('ggplot')
    gs = gridspec.GridSpec(3, 3,
                           width_ratios=[1, 5, 1],
                           height_ratios=[1, 5, 1]
                           )

    m_levels = np.arange(-100, 100, 0.5)

    plt.subplot(gs[0, 0]).contour(f1[ds + "/PML_6"][:, :, 0, o_dir, step], levels=m_levels)
    plt.subplot(gs[0, 1]).contour(f1[ds + "/PML_0"][:, :, 0, o_dir, step], levels=m_levels)
    plt.subplot(gs[0, 2]).contour(f1[ds + "/PML_7"][:, :, 0, o_dir, step], levels=m_levels)
    plt.subplot(gs[1, 0]).contour(f1[ds + "/PML_2"][:, :, 0, o_dir, step], levels=m_levels)
    plt.subplot(gs[1, 1]).contour(f1[ds + "/Center"][:, :, 0, o_dir, step], levels=m_levels)
    plt.subplot(gs[1, 2]).contour(f1[ds + "/PML_5"][:, :, 0, o_dir, step], levels=m_levels)
    plt.subplot(gs[2, 0]).contour(f1[ds + "/PML_3"][:, :, 0, o_dir, step], levels=m_levels)
    plt.subplot(gs[2, 1]).contour(f1[ds + "/PML_1"][:, :, 0, o_dir, step], levels=m_levels)
    plt.subplot(gs[2, 2]).contour(f1[ds + "/PML_4"][:, :, 0, o_dir, step], levels=m_levels)

    for i, ax in enumerate(plt.gcf().axes):
        # ax.text(0.5, 0.5, "ax%d" % (i+1), va="center", ha="center")
        # for tl in ax.get_xticklabels() + ax.get_yticklabels():
        #     tl.set_visible(False)
        plt.setp(ax, xticks=[], yticks=[])
else:
    plt.contour(f1[ds + "/Center"][:, :, 0, o_dir, step])

plt.show()


# center = make_axes_locatable(axes)
#
# ax_top = center.append_axes("top", 1.2, pad=0.1, sharex=axes)
# ax_bottom = center.append_axes("bottom", 1.2, pad=0.1, sharex=axes)
#
# ax_left = center.append_axes("left", 1.2, pad=0.1, sharey=axes)
# ax_right = center.append_axes("right", 1.2, pad=0.1, sharey=axes)
#
# plt.setp(
#     ax_left.get_xticklabels() + ax_left.get_yticklabels() + ax_right.get_xticklabels() + ax_right.get_yticklabels(),
#     visible=False)
#
# ax_top.imshow(f1["/checkpoint/E/PML_0"][:, :, 0, 2, step])
# ax_bottom.imshow(f1["/checkpoint/E/PML_1"][:, :, 0, 2, step])
#
# ax_left.imshow(f1["/checkpoint/E/PML_2"][:, :, 0, 2, step])
# axes.imshow(f1["/checkpoint/E/Center"][:, :, 0, 2, step])
# ax_right.imshow(f1["/checkpoint/E/PML_5"][:, :, 0, 2, step])
