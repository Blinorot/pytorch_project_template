import io

import matplotlib.pyplot as plt
import PIL
from torchvision.transforms import ToTensor

plt.switch_backend("agg")  # fix RuntimeError: main thread is not in main loop


def plot_images(imgs, writer, config):
    names = config.writer.names
    figsize = config.writer.figsize
    fig, axes = plt.subplots(1, len(names), figsize=figsize)
    for i in range(len(names)):
        img = imgs[i].permute(1, 2, 0)
        axes[i].imshow(img)  # channels must be in the last dim
        axes[i].set_title(names[i])
        axes[i].axis("off")
    buf = io.BytesIO()
    fig.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    image = ToTensor()(PIL.Image.open(buf))

    plt.close()

    return image
