import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt


def show_images(imgs, figsize=(20, 10), rows=1, cmap="gray"):
    """ Show images in grid """
    plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(rows + 1, int(len(imgs.items()) / rows))

    for (index, (name, img)) in enumerate(imgs.items()):
        ax = plt.subplot(gs[index])
        ax.set_title(name)
        ax.axis("off")
        ax.imshow(img, cmap=cmap)
