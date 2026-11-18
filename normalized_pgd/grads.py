"""Gradient of the prediction w.r.t. the input pixels, overlaid on the image.
Importance = sum over colour channels of |d softmax_pred / d x| -- eq1"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

torch.backends.cudnn.allow_tf32 = False  # full fp32 convs: TF32 rounding visibly changes gradient maps


def grad_map(model, x):
    """x: B x C x H x W in [0, 1]  ->  predicted class, its probability, importance B x H x W in [0, 1]"""
    x = x.clone().requires_grad_(True)
    prob, pred = model(x).double().softmax(1).max(1)  # model(x) runs the model and gives logits -convert to -> float64, totla prob = 1, max(1) gives for each img the largest prob and prediction
    prob.sum().backward()  # each prob depends only on its own image
    imp = x.grad.abs().sum(1) #eq 1
    scale = imp.flatten(1).quantile(0.99, dim=1)[:, None, None]
    imp = imp / torch.where(scale > 0, scale, torch.ones_like(scale))  # preserve tiny gradients; guard zero only
    return pred, prob.detach(), imp.clamp(0, 1)


def show(ax, img, title):
    ax.imshow(img.detach().cpu().permute(1, 2, 0).squeeze(), cmap="gray", vmin=0, vmax=1)  # cpu -> matplotlib cant read gpu mem so move to cpu,permute -> convert dims from c h w to h w c (what matplotlib wants) ,squeeze -> gray only if 1 channel
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def plot(heads, x, y, classes, path, title):
    """One row per image: image | for each head: |grad| map, image x |grad|"""
    n, k = len(x), len(heads)
    fig, axes = plt.subplots(n, 1 + 2 * k, figsize=(2 * (1 + 2 * k), 2.2 * n), squeeze=False, layout="constrained")
    for i in range(n):
        show(axes[i, 0], x[i], f"true: {classes[y[i]]}")
    for j, (name, model) in enumerate(heads.items()):
        pred, prob, imp = grad_map(model, x)
        for i in range(n):
            show(axes[i, 1 + 2 * j], imp[i][None], f"{name} |grad|")
            show(axes[i, 2 + 2 * j], x[i] * imp[i], f"{name}: {classes[pred[i]]} ({prob[i]:.2f})")
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)
