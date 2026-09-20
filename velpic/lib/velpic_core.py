"""
velpic_core.py  --  shared machinery for every VeLPIC experiment.

One place for: loading a model and its data, pulling penultimate activations,
building VeLPIC class vectors, swapping the pre-softmax layer, and drawing the
gradient-map figure. main/ and extras/ both import this, so the VeLPIC maths is
defined exactly once.

The VeLPIC maths itself is NOT implemented here. It is called from the authors
repo (Ketha & Ramaswamy, arXiv:2603.19865), which is never modified:

    classwise_data    MASC/subspace_pytorch.py   group activations by label
    Added_data_layer  MASC/subspace_pytorch.py   augment with negatives
    PCA_new_layer     MASC/scratch_pca.py        first principal component
    avg_neg           MAVC/MASV_pytorch.py       which vectors to flip
    accuracy_model    VeLPIC_Weight_change.py    accuracy loop

Gradients and plotting come from visualize_prediction_gradients.py, also
unmodified.
"""

import contextlib
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]          # .../visgrads
REPO = ROOT.parent / "Dynamics_during_training_DNN" / "VeLPIC_during_training"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import visualize_prediction_gradients as vpg        # noqa: E402

IMAGENETTE = ROOT / "data" / "imagenette2-320"


# ----------------------------------------------------------------------------
# The authors' code
# ----------------------------------------------------------------------------
def import_repo():
    """Import the authors' modules. Their repo is never modified."""
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from MASC import subspace_pytorch as subspace
    from MASC import scratch_pca as sp              # the numpy one angle_pytorch.py calls
    from MAVC import MASV_pytorch as mavc
    import VeLPIC_Weight_change as vwc
    return subspace, sp, mavc, vwc


# ----------------------------------------------------------------------------
# Locating and editing the pre-softmax layer
# ----------------------------------------------------------------------------
def last_linear(model):
    """The pre-softmax layer: the final nn.Linear in forward order.

    Found by traversal, not by attribute name, so the same code works for
    WideResNet (.fc), ResNet-50 (.fc) and SmallCNN (.classifier[3]).
    """
    found = [m for _, m in model.named_modules() if isinstance(m, nn.Linear)]
    if not found:
        raise RuntimeError("no nn.Linear found")
    return found[-1]


def replace_last_linear(model, new_layer):
    name = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    parts = name.split(".")
    owner = model
    for p in parts[:-1]:
        owner = getattr(owner, p)
    setattr(owner, parts[-1], new_layer)
    return model


def set_head(layer, W):
    """Change the last layer's weights. This is the whole intervention.

    The bias is left alone: it has no effect on the input gradient, since
    d/dx [w.x + b] = w^T J(x) and the constant drops out. Verified empirically
    at cosine 1.000000 between the zeroed and kept versions.
    """
    with torch.no_grad():
        layer.weight.copy_(W.to(layer.weight.device, layer.weight.dtype))


@torch.no_grad()
def penultimate(model, layer, loader, device):
    """Activations entering `layer` -- the last hidden representation."""
    box, xs, ys = {}, [], []
    h = layer.register_forward_hook(lambda m, inp, out: box.__setitem__("x", inp[0].detach()))
    try:
        for x, y in loader:
            model(x.to(device))
            xs.append(box["x"].float().cpu())
            ys.append(y)
    finally:
        h.remove()
    return torch.cat(xs), torch.cat(ys)


@torch.no_grad()
def accuracy(model, loader, device):
    c = t = 0
    for x, y in loader:
        c += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        t += y.numel()
    return 100.0 * c / t


# ----------------------------------------------------------------------------
# VeLPIC, via the authors' functions
# ----------------------------------------------------------------------------
def velpic_vectors(X, y, num_class, verbose=True):
    """Algorithm 1 of the paper. Returns (W, explained_variance, cos_to_mean)."""
    subspace, sp, mavc, _ = import_repo()
    obj = subspace.classwise_data(X, y, number_class=num_class)

    P, var = [], []
    for c in range(num_class):
        Xc = obj[f"class{c}"]
        Xa = subspace.Added_data_layer(Xc)              # x and -x
        vec, ev = sp.PCA_new_layer(Xa.numpy(), 1)       # PC1, unit norm
        P.append(torch.as_tensor(np.asarray(vec), dtype=torch.float32))
        var.append(float(ev))
    P = torch.cat(P, dim=0)

    # class-major layout, matching what dot_part_train builds
    class_dot = (X @ P.T).T.reshape(-1, 1)
    flip = mavc.avg_neg(class_dot, X.shape[0], num_class=num_class)
    W = torch.stack([-P[c] if flip[c] else P[c] for c in range(num_class)])

    mu = torch.stack([obj[f"class{c}"].mean(0) for c in range(num_class)])
    mu = mu / mu.norm(dim=1, keepdim=True)
    cos = (W * mu).sum(1)

    if verbose:
        print("\n  class   explVar(PC1)   flipped   cos(v_j, class mean)")
        for c in range(num_class):
            print(f"   {c:>3}       {var[c]:.3f}        {str(flip[c]):>5}        {cos[c]:+.4f}")
    return W, var, cos


def build_heads(names, W_orig, W_velpic, X, y, num_class, seed=0):
    """Weight matrices for each head, all (M, d)."""
    out = {}
    for n in names:
        if n == "orig":
            out[n] = W_orig.clone()
        elif n == "velpic":                                   # head 1
            out[n] = W_velpic.clone()
        elif n == "resid":                                    # head 2, as on the whiteboard
            out[n] = W_orig - W_velpic
        elif n == "resid_orth":                               # scale-free variant
            out[n] = W_orig - (W_orig * W_velpic).sum(1, keepdim=True) * W_velpic
        elif n == "rand":                                     # control
            g = torch.Generator().manual_seed(seed)
            R = torch.randn(W_velpic.shape, generator=g)
            out[n] = R / R.norm(dim=1, keepdim=True)
        elif n == "classmean":                                # expected to match velpic
            Mu = torch.stack([X[y == c].mean(0) for c in range(num_class)])
            out[n] = Mu / Mu.norm(dim=1, keepdim=True)
        else:
            raise ValueError(f"unknown head: {n}")
    return out


def participation_ratio(p):
    """Effective number of pixels carrying the importance.
    Low = concentrated and structured. High = diffuse noise."""
    p = p.ravel()
    return float(p.sum() ** 2 / (p ** 2).sum())


# ----------------------------------------------------------------------------
# Datasets and models
# ----------------------------------------------------------------------------
def imagenet_indices(wnids):
    """Imagenette wnids -> ImageNet-1k class indices.

    ImageNet-1k ordering is the 1000 wnids sorted alphabetically; the
    imagenet-sample-images folder has one file per class named <wnid>_<label>,
    so sorting those recovers the ordering without the full dataset.
    """
    sample = ROOT / "imagenet-sample-images"
    files = sorted(p.name.split("_")[0] for p in sample.glob("n*.JPEG"))
    lookup = {w: i for i, w in enumerate(files)}
    names = [p.name.split("_", 1)[1].replace(".JPEG", "").replace("_", " ")
             for p in sorted(sample.glob("n*.JPEG"))]
    idx = [lookup[w] for w in wnids]
    return idx, [names[i] for i in idx]


def load_setup(dataset, model_spec, device, batch_size=128, workers=6, num_train=None):
    """Returns (model, layer, train_loader, test_loader, class_names, num_class)."""
    if dataset == "imagenet":
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                 transforms.ToTensor()])
        train = datasets.ImageFolder(IMAGENETTE / "train", transform=tf)
        test = datasets.ImageFolder(IMAGENETTE / "val", transform=tf)
        idx, names = imagenet_indices(train.classes)

        ns = _ns(dataset, device, batch_size)
        with _at_root():
            model, _ = vpg.load_model(model_spec, ns)
        model.to(device).eval()

        full = last_linear(model)
        sliced = nn.Linear(full.in_features, len(idx), bias=full.bias is not None)
        with torch.no_grad():
            sliced.weight.copy_(full.weight[idx])
            if full.bias is not None:
                sliced.bias.copy_(full.bias[idx])
        replace_last_linear(model, sliced.to(device))
        model.eval()
        num_class = len(idx)
    else:
        train = vpg.small_dataset(dataset, str(ROOT / "data"), train=True)
        test = vpg.small_dataset(dataset, str(ROOT / "data"), train=False)
        names = list(test.classes)
        ns = _ns(dataset, device, batch_size)
        with _at_root():
            model, _ = vpg.load_model(model_spec, ns)
        model.to(device).eval()
        num_class = 10

    if num_train:
        train = Subset(train, range(min(num_train, len(train))))
    tl = DataLoader(train, batch_size=batch_size, num_workers=workers)
    vl = DataLoader(test, batch_size=batch_size, num_workers=workers)
    return model, last_linear(model), tl, vl, names, num_class


def _ns(dataset, device, batch_size):
    import argparse
    return argparse.Namespace(dataset=dataset, data_root=str(ROOT / "data"), device=device,
                              epochs=3, batch_size=batch_size,
                              rb_model_dir=str(ROOT / "models"))


@contextlib.contextmanager
def _at_root():
    """Run with the project root as cwd.

    load_smallcnn in visualize_prediction_gradients.py resolves its checkpoint
    as the relative Path("saved_models"). Scripts here run from their own
    folders, so without this it silently fails to find the existing checkpoint
    and trains a throwaway model instead. That file is not modified, so the fix
    belongs at the call site.
    """
    prev = os.getcwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(prev)


def sample_images(dataset, n):
    """n evenly spaced test images as (tensor CxHxW, label)."""
    if dataset == "imagenet":
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                 transforms.ToTensor()])
        ds = datasets.ImageFolder(IMAGENETTE / "val", transform=tf)
    else:
        ds = vpg.small_dataset(dataset, str(ROOT / "data"), train=False)
    step = max(1, len(ds) // n)
    return [ds[i * step] for i in range(n)]


# ----------------------------------------------------------------------------
# Gradient maps
# ----------------------------------------------------------------------------
def gradient_rows(model, layer, heads, W_orig, picks, class_names, device, score="logit"):
    """One row per image; each row holds one cell per head."""
    rows, prs = [], {n: [] for n in heads}
    for xt, yt in picks:
        xb = xt.unsqueeze(0)
        img = vpg.to_numpy_image(xb)
        x = xb.to(device)
        row = {"image": img, "true": class_names[int(yt)], "cells": []}
        for name, W in heads.items():
            set_head(layer, W)
            pred, prob, _, imp = vpg.prediction_gradient(model, x.clone(), score_type=score)
            prs[name].append(participation_ratio(imp))
            pn = vpg.normalize_importance(imp, 99)
            row["cells"].append((name, class_names[int(pred)], float(prob), pn,
                                 vpg.make_overlay(img, pn)))
        rows.append(row)
    set_head(layer, W_orig)
    return rows, {n: float(np.mean(v)) for n, v in prs.items()}


def draw(rows, title, path):
    heads = [c[0] for c in rows[0]["cells"]]
    ncol = 1 + 2 * len(heads)
    fig, ax = plt.subplots(len(rows), ncol,
                           figsize=(2.1 * ncol, 2.3 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        cmap = "gray" if row["image"].ndim == 2 else None
        ax[r][0].imshow(row["image"], cmap=cmap)
        if r == 0:
            ax[r][0].set_title("input", fontsize=9)
        ax[r][0].set_ylabel(row["true"], fontsize=6)
        for i, (name, pred, prob, pn, ov) in enumerate(row["cells"]):
            a1, a2 = ax[r][1 + 2 * i], ax[r][2 + 2 * i]
            a1.imshow(pn, cmap="gray")
            a2.imshow(ov, cmap=cmap)
            if r == 0:
                a1.set_title(f"{name}\n|grad|", fontsize=8)
                a2.set_title(f"{name}\noverlay", fontsize=8)
            a2.set_xlabel(f"{pred} {prob:.2f}", fontsize=5)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
