"""
velpic_memorize.py  --  the paper's own regime: train on scrambled labels.

Everything in ../../main runs on clean, pre-trained models. That is OUTSIDE the
setting the paper studies. The paper trains its own networks on deliberately
corrupted labels, so the model memorizes noise and its test accuracy collapses;
VeLPIC then recovers part of the lost generalization. This script reproduces
that, which requires training our own model -- no pre-trained network exists
that was trained on scrambled labels.

  1. randomise a fraction of the training labels
  2. train until the network MEMORIZES them (train acc -> ~100% on the
     corrupted labels, test acc on true labels collapses)
  3. build VeLPIC vectors from the CORRUPTED label groups, as the paper does
  4. swap them into the pre-softmax layer and see whether test accuracy on TRUE
     labels jumps -- the paper's Section 8 claim
  5. same three heads as main: orig, velpic, resid

Feasibility note. CIFAR-10 trains from scratch in minutes. Real 1000-class
ImageNet is not feasible here (150 GB of data we do not have, and weeks of GPU
time), so the ImageNet-resolution version fine-tunes a pre-trained ResNet-50 on
corrupted Imagenette at 224x224. That is a memorization experiment at ImageNet
resolution, not on full ImageNet -- the backbone starts from pre-trained
features rather than from scratch, unlike the paper.

  python velpic_memorize.py --dataset cifar10  --corrupt 0.4 --epochs 80
  python velpic_memorize.py --dataset imagenet --corrupt 0.4 --epochs 12
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import velpic_core as vc                                  # noqa: E402

HEADS = ["orig", "velpic", "resid"]


class Corrupted(Dataset):
    """A dataset with a fraction of labels replaced by uniformly random ones.

    Keeps both label sets: `corrupt` is what the model trains on, `true` is what
    generalization is measured against.
    """

    def __init__(self, base, n, prob, seed, num_class=10):
        self.base, self.n = base, n
        g = np.random.RandomState(seed)
        # read labels from the index where possible; indexing base[i] would
        # decode every image just to get its label (slow for ImageFolder)
        if hasattr(base, "targets"):
            self.true = np.asarray(base.targets[:n], dtype=int)
        else:
            self.true = np.array([int(base[i][1]) for i in range(n)])
        self.corrupt = self.true.copy()
        hit = g.rand(n) < prob
        self.corrupt[hit] = g.randint(0, num_class, hit.sum())
        self.n_flipped = int((self.corrupt != self.true).sum())

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x, _ = self.base[i]
        return x, int(self.corrupt[i])


def build_model(dataset, device):
    """A network to train from scratch (CIFAR) or fine-tune (ImageNet)."""
    if dataset == "imagenet":
        from torchvision import models
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        net.fc = nn.Linear(net.fc.in_features, 10)
        model = vc.vpg.WithNormalization(net, vc.vpg.IM_MEAN, vc.vpg.IM_STD)
    else:
        ch = 1 if dataset in ("mnist", "fashion_mnist") else 3
        mean, std = ((vc.vpg.MNIST_MEAN, vc.vpg.MNIST_STD) if ch == 1
                     else (vc.vpg.CIFAR_MEAN, vc.vpg.CIFAR_STD))
        model = vc.vpg.WithNormalization(vc.vpg.SmallCNN(ch, 10), mean, std)
    return model.to(device)


def load_base(dataset):
    if dataset == "imagenet":
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                 transforms.ToTensor()])
        return (datasets.ImageFolder(vc.IMAGENETTE / "train", transform=tf),
                datasets.ImageFolder(vc.IMAGENETTE / "val", transform=tf))
    return (vc.vpg.small_dataset(dataset, str(vc.ROOT / "data"), train=True),
            vc.vpg.small_dataset(dataset, str(vc.ROOT / "data"), train=False))


def train_memorize(model, ds, device, epochs, bs, lr, test_loader, workers):
    """No augmentation, so the network can memorize the noise."""
    # persistent_workers matters a lot here: without it Windows respawns every
    # worker process at the start of each epoch, which for 80 epochs costs far
    # more than the training itself and leaves the GPU idle.
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=workers,
                        persistent_workers=workers > 0)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    every = max(1, epochs // 8)
    model.train()
    for ep in range(1, epochs + 1):
        c = t = 0
        for x, yy in loader:
            x, yy = x.to(device), yy.to(device)
            opt.zero_grad(set_to_none=True)
            o = model(x)
            nn.functional.cross_entropy(o, yy).backward()
            opt.step()
            c += (o.argmax(1) == yy).sum().item()
            t += yy.numel()
        if ep % every == 0 or ep == epochs:
            model.eval()
            te = vc.accuracy(model, test_loader, device)
            model.train()
            print(f"    epoch {ep:>3}   train(corrupt) {100*c/t:6.2f}%   test(true) {te:6.2f}%")
    model.eval()
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar10",
                   choices=["mnist", "fashion_mnist", "cifar10", "imagenet"])
    p.add_argument("--corrupt", type=float, default=0.4)
    p.add_argument("--num_train", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--score", choices=["logit", "softmax"], default="logit")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "outputs"))
    args = p.parse_args()

    _, _, _, vwc = vc.import_repo()
    device = vwc.dev
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # MNIST and CIFAR live in memory as tensors, so worker processes are pure
    # overhead there. Only ImageFolder (ImageNet) benefits from parallel decode.
    workers = args.workers if args.dataset == "imagenet" else 0
    print(f"device : {device}\ndataset: {args.dataset}   dataloader workers: {workers}")

    base_train, base_test = load_base(args.dataset)
    n = min(args.num_train, len(base_train))
    ds = Corrupted(base_train, n, args.corrupt, args.seed)
    print(f"\n[1] {n} train images, {ds.n_flipped} labels flipped "
          f"({100*ds.n_flipped/n:.1f}%)")

    test_loader = DataLoader(base_test, batch_size=args.batch_size, num_workers=workers)
    true_loader = DataLoader(Subset(base_train, range(n)), batch_size=args.batch_size,
                             num_workers=workers)
    corrupt_loader = DataLoader(ds, batch_size=args.batch_size, num_workers=workers)

    tag = f"{args.dataset}_c{args.corrupt}_n{n}_e{args.epochs}"
    ckpt = out / f"memorized_{tag}.pth"
    model = build_model(args.dataset, device)
    if ckpt.exists():
        print(f"\n[2] loading {ckpt.name}")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
    else:
        print(f"\n[2] training to memorize ({args.epochs} epochs, no augmentation)")
        train_memorize(model, ds, device, args.epochs, args.batch_size, args.lr,
                       test_loader, workers)
        torch.save(model.state_dict(), ckpt)
        print(f"    saved {ckpt.name}")

    layer = vc.last_linear(model)
    a_corrupt = vc.accuracy(model, corrupt_loader, device)
    a_train = vc.accuracy(model, true_loader, device)
    a_test = vc.accuracy(model, test_loader, device)
    print(f"\n    train acc on CORRUPTED labels : {a_corrupt:6.2f}%  (memorization: want ~100)")
    print(f"    train acc on TRUE labels      : {a_train:6.2f}%")
    print(f"    test  acc on TRUE labels      : {a_test:6.2f}%")

    # --- VeLPIC from the corrupted groups, as the paper does ----------------
    print("\n[3] VeLPIC vectors from CORRUPTED-label groups")
    X, y = vc.penultimate(model, layer, corrupt_loader, device)
    W_velpic, _, _ = vc.velpic_vectors(X, y, 10)
    W_orig = layer.weight.detach().float().cpu().clone()
    heads = vc.build_heads(HEADS, W_orig, W_velpic, X, y, 10)

    print("\n[4] weight intervention -- test accuracy on TRUE labels")
    accs = {}
    for name, W in heads.items():
        vc.set_head(layer, W)
        accs[name] = vc.accuracy(model, test_loader, device)
        print(f"      {name:<8} {accs[name]:6.2f}%")
    vc.set_head(layer, W_orig)
    print(f"\n    >>> transfer effect: {a_test:.2f}% -> {accs['velpic']:.2f}%  "
          f"({accs['velpic'] - a_test:+.2f} points)")

    # --- gradient maps ------------------------------------------------------
    print(f"\n[5] gradient maps for {args.num_images} test images")
    names = (vc.imagenet_indices(base_train.classes)[1] if args.dataset == "imagenet"
             else list(base_test.classes))
    picks = vc.sample_images(args.dataset, args.num_images)
    rows, prs = vc.gradient_rows(model, layer, heads, W_orig, picks, names,
                                 device, args.score)
    for nme, v in prs.items():
        print(f"      {nme:<8} participation ratio {v:9.1f}")

    path = vc.draw(rows, f"memorized {args.dataset}, {int(100*args.corrupt)}% corrupt "
                         f"(model {a_test:.1f}% -> VeLPIC {accs['velpic']:.1f}%)",
                   out / f"velpic_memorized_{tag}.png")
    with open(out / f"results_{tag}.csv", "w") as f:
        f.write("quantity,value\n")
        f.write(f"train_acc_corrupt_labels,{a_corrupt}\ntrain_acc_true_labels,{a_train}\n")
        f.write(f"model_test_acc,{a_test}\n")
        for nme in heads:
            f.write(f"{nme}_test_acc,{accs[nme]}\n{nme}_participation,{prs[nme]}\n")
    print(f"\nDONE\nsaved: {path}")


if __name__ == "__main__":
    main()
