import argparse
import copy
import sys
import types
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms as T

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "VeLPIC_during_training"))            # original velpic code
import VeLPIC_Weight_change as vwc                                  # weight_change, pca_load_mavc, accuracy_model
from MAVC import MASV_pytorch as mavc                               # dot_part_train, avg_neg
from MASC import subspace_pytorch as subspace, scratch_pca as sp    # classwise_data, Added_data_layer, PCA_new_layer
from CNN_code import cnn_create                                     # NgnCnn + its optimiser settings
from robustbench.utils import load_model
import grads

DATA, MODELS, OUT = ROOT / "data", ROOT / "models", ROOT / "outputs"
IMAGENETTE_IDX = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]  # ImageNet ids of the 10 Imagenette classes


def train(model, ds, ckpt, epochs):
    # for MNIST, trains existing code's CNN with their optimizer and batch size
    loss_fn, opt, bs = cnn_create.model_params("CNN", "MNIST", model)
    model.train()
    for ep in range(epochs):
        for x, y in DataLoader(ds, bs, shuffle=True):
            # x: images (128   *   1 * 28 * 28), y: true digits (128 nos)
            opt.zero_grad()
            loss = loss_fn(model(x.cuda()), y.cuda())
            loss.backward() # computes gradient of loss wrt every weight
            opt.step()
        print(f"epoch {ep + 1}/{epochs}  last batch loss {loss.item():.4f}")
    ckpt.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ckpt)


def load(name, epochs):
    # model on GPU taking [0, 1] images, train set, test set. Everything downloads on first run.
    tf = T.ToTensor() # converts a picture into tensor of shape C x H x W with vals in [0, 1] bcoz divide by 255
    if name == "mnist":
        tr, te = (datasets.MNIST(DATA, train=t, download=True, transform=tf) for t in (True, False))
        # Normalize((0.5,), (0.5,)) moved inside the model, so gradients are w.r.t. raw pixels
        model = nn.Sequential(T.Normalize((0.5,), (0.5,)), cnn_create.model_create("CNN", "MNIST")).cuda()
        ckpt = MODELS / "mnist_cnn.pth"
        if not ckpt.exists():
            train(model, tr, ckpt, epochs)
        model.load_state_dict(torch.load(ckpt))
    elif name in ("cifar10", "cifar10r"):
        tr, te = (datasets.CIFAR10(DATA, train=t, download=True, transform=tf) for t in (True, False))
        # RobustBench WRN-28-10, takes [0, 1] images: cifar10 = normally trained, cifar10r = adversarially trained (Linf)
        rb_name = "Standard" if name == "cifar10" else "Carmon2019Unlabeled"
        model = load_model(rb_name, model_dir=MODELS, dataset="cifar10", threat_model="Linf").cuda()
    else:
        tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
        tr, te = (datasets.Imagenette(DATA, split=s, size="320px", download=True, transform=tf) for s in ("train", "val"))
        net = models.resnet50(weights="IMAGENET1K_V2")
        fc, net.fc = net.fc, nn.Linear(2048, 10)
        net.fc.load_state_dict({k: v[IMAGENETTE_IDX] for k, v in fc.state_dict().items()})  # W: 1000x2048 -> 10x2048
        model = nn.Sequential(T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), net).cuda()
    return model.eval(), tr, te


def head(model):
    """The pre-softmax layer (last nn.Linear)."""
    return [m for m in model.modules() if isinstance(m, nn.Linear)][-1]


@torch.no_grad()
def features(model, ds, bs):
    """Inputs of the pre-softmax layer (N x d, on GPU) and labels."""
    feats, ys = [], []
    hook = head(model).register_forward_hook(lambda m, inp, out: feats.append(inp[0]))
    for x, y in DataLoader(ds, bs):
        model(x.cuda())
        ys.append(y)
    hook.remove()
    return torch.cat(feats), torch.cat(ys)


def velpic(feats, ys, C, folder):
    """per class, top PC of [x; -x], sign-flipped so the class projects positively -> V (C x d).
    Their functions pass PCs through files: <folder>/Run_1/0/pca_train_corrupt_last_<class>.pt"""
    (folder / "Run_1" / "0").mkdir(parents=True, exist_ok=True)
    obj = subspace.classwise_data(feats, ys, number_class=C)              # split features by class
    for c in range(C):
        X = subspace.Added_data_layer(obj[f"class{c}"])                    # add -x for each x: PCA through the origin
        pc, _ = sp.PCA_new_layer(X.cpu(), 1)                                # PC1, unit length
        torch.save(torch.from_numpy(pc), folder / "Run_1" / "0" / f"pca_train_corrupt_last_{c}.pt")
    dots = mavc.dot_part_train("last", folder, 0, 1, "corrupt", feats, C)  # every train point on every PC
    neg = mavc.avg_neg(dots, len(feats), num_class=C)                     # mean projection < 0 -> flip that PC
    return vwc.pca_load_mavc(folder, 1, 0, "corrupt", "last", neg, C)     # stacked (flipped) PCs: C x d


def swap(model, W):
    """Copy of model with pre-softmax weights = W and bias = 0, via the weight_change()."""
    m = copy.deepcopy(model)
    fc = head(m)
    # weight_change() edits `.fc3`; its 'AlexNet' branch zeroes the bias, 'CNN' is for a layer with no bias
    vwc.weight_change(types.SimpleNamespace(fc3=fc), W, withbias=False,
                      type_network="CNN" if fc.bias is None else "AlexNet")
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=["mnist", "cifar10", "cifar10r", "imagenet"])
    p.add_argument("--n_show", type=int, default=8, help="test images in the gradient figure")
    p.add_argument("--epochs", type=int, default=5, help="MNIST CNN, only if models/mnist_cnn.pth is missing")
    p.add_argument("--bs", type=int, default=128)
    a = p.parse_args()
    out = OUT / a.dataset
    out.mkdir(parents=True, exist_ok=True)

    model, tr, te = load(a.dataset, a.epochs)
    classes = [c[0] if isinstance(c, tuple) else c for c in te.classes]  # Imagenette names are tuples

    # 1. features entering the pre-softmax layer, for every training image
    feats, ys = features(model, tr, a.bs)

    # 2. VeLPIC vectors V, and the trained weights W
    V = velpic(feats, ys, len(classes), out / "pca") # Velpic matrix of size 10 x d
    W = head(model).weight.detach() # last layer trained weights matrix (10 x d)

    # 3. three heads on the same network (a dict of 3 models)
    heads = {"orig": model, "velpic": swap(model, V), "resid": swap(model, W - V)}

    # 4. test accuracy of each head
    with torch.no_grad():
        acc = {k: vwc.accuracy_model(m, DataLoader(te, a.bs), f"{a.dataset} {k} test") for k, m in heads.items()}
    (out / "accuracy.csv").write_text("head,test_acc\n" + "".join(f"{k},{v:.2f}\n" for k, v in acc.items()))

    # 5. gradient of each head's prediction w.r.t. the pixels, on n_show random test images
    x, y = next(iter(DataLoader(te, a.n_show, shuffle=True, generator=torch.Generator().manual_seed(0))))
    grads.plot(heads, x.cuda(), y, classes, out / "grads.png", f"{a.dataset}: gradient of the prediction per head")
    print(f"saved {out / 'accuracy.csv'}, {out / 'grads.png'}")


if __name__ == "__main__":
    main()
