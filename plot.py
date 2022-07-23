import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec


class Recorder:

    def __init__(self, name, fps=1):
        self.frames = []
        self.name = name
        self.fps = fps

    def add_frame(self, fig, save=False):
        # fig.tight_layout(pad=0)
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(image_from_plot)
        if save:
            self.save_gif()

    def save_gif(self):
        imageio.mimsave(self.name, self.frames, fps=self.fps)


def plot_filters(layout, title=None, recorder=None):
    cols = max([l[1][1] for l in layout])
    rows = sum([l[1][0] for l in layout]) + len(layout)

    hr = [1]
    shrink = 0
    for i in range(len(layout)):
        for ir in range(layout[i][1][0]):
            hr.append(5)
        hr.append(1)
        shrink += 1

    fig, ax = plt.subplots(figsize=(cols, rows - shrink + 1), nrows=rows, ncols=cols,
                           gridspec_kw={'height_ratios': hr[:-1]}, dpi=200)
    for i in range(rows):
        for j in range(cols):
            ax[i, j].set_aspect("equal")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].set_axis_off()

    cmap = 'bwr'
    all_filters = [l[0].get_kernels().cpu().detach().numpy() for l in layout]
    clim = max(np.abs(min([np.min(f) for f in all_filters])), np.abs(max([np.max(f) for f in all_filters])))
    hr = []
    row = 1
    for i, f in enumerate(all_filters):
        filters = np.reshape(f, (layout[i][0].n_filters, layout[i][0].rotations, f.shape[2], f.shape[3]))
        count = 0
        for ir in range(layout[i][1][0]):
            hr.append(4)
            for ic in range(layout[i][1][1]):
                if count % layout[i][0].rotations == 0:
                    ax[row, ic].set_axis_on()
                fr = int(count / layout[i][0].rotations)
                fc = count % layout[i][0].rotations
                # print(i, fr, fc, "->", row, ic)
                img = filters[fr, fc]
                ax[row, ic].imshow(img, cmap=cmap, vmin=-clim, vmax=clim, origin='lower')
                count += 1
            row += 1

        row += 1
        hr.append(1)

    if title is not None:
        plt.suptitle(title, y=1, size=20)
    gs = gridspec.GridSpec(rows, cols)
    gs.update(left=0.05, right=0.95, top=0.965, bottom=0.03, wspace=0.02, hspace=0.02)
    if recorder is not None:
        recorder.add_frame(fig, save=True)
    plt.show()
    plt.close(fig)