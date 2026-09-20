"""
velpic_pgd.py  --  does the VeLPIC head actually resist attack?

Accuracy and gradient maps both failed to separate the heads. Robustness is the
third axis, and the one the robust/non-robust hypothesis is really about: if the
VeLPIC vectors capture "robust features", the VeLPIC head should survive a
perturbation that breaks the original head.

Two attack settings, answering different questions:

  white-box  attack each head on its own terms, each defended against its own
             worst case. The honest comparison.
  transfer   craft the perturbation against the ORIGINAL head and feed it to
             every head. Weaker, but it is what a shared-input figure needs.

Reference points matter, so run the robust model too. At eps=8/255 a standard
model goes to ~0% and an adversarially trained one holds ~60%. Those bracket the
scale any result has to be read against.

  python velpic_pgd.py --dataset cifar10
  python velpic_pgd.py --dataset cifar10 --robust
  python velpic_pgd.py --dataset cifar10 --ckpt ../memorization/outputs/memorized_cifar10_c0.4_n10000_e80.pth --corrupt 0.4
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
sys.path.insert(0, str(HERE.parent / "memorization"))
import velpic_core as vc                                  # noqa: E402
from velpic_memorize import Corrupted, build_model        # noqa: E402

STANDARD = {"cifar10": "robustbench:cifar10:Standard",
            "imagenet": "robustbench:imagenet:Standard_R50",
            "mnist": "smallcnn"}
ROBUST = {"cifar10": "robustbench:cifar10:Carmon2019Unlabeled",
          "imagenet": "robustbench:imagenet:Salman2020Do_R50"}
HEADS = ["orig", "velpic", "resid", "resid_orth"]


def pgd(model, x, y, eps, alpha, steps):
    """PGD-Linf with a random start. Inputs live in [0, 1]."""
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = (x + delta).clamp(0, 1) - x
    for _ in range(steps):
        delta.requires_grad_(True)
        loss = F.cross_entropy(model(x + delta), y)
        (g,) = torch.autograd.grad(loss, delta)
        delta = (delta.detach() + alpha * g.sign()).clamp(-eps, eps)
        delta = (x + delta).clamp(0, 1) - x
    return (x + delta).detach()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="cifar10", choices=["mnist", "cifar10", "imagenet"])
    p.add_argument("--model", default=None)
    p.add_argument("--robust", action="store_true")
    p.add_argument("--ckpt", default=None, help="a memorized checkpoint to attack instead")
    p.add_argument("--corrupt", type=float, default=None,
                   help="with --ckpt: build the vectors from these corrupted labels")
    p.add_argument("--num_train", type=int, default=10000)
    p.add_argument("--num_eval", type=int, default=2000)
    p.add_argument("--eps", type=float, default=8 / 255)
    p.add_argument("--alpha", type=float, default=2 / 255)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "outputs"))
    args = p.parse_args()

    _, _, _, vwc = vc.import_repo()
    device = vwc.dev
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {device}   eps={args.eps*255:.0f}/255  steps={args.steps}  n={args.num_eval}")

    if args.ckpt:
        model = build_model(args.dataset, device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
        model.eval()
        tag = Path(args.ckpt).stem
        base = vc.vpg.small_dataset(args.dataset, str(vc.ROOT / "data"), train=True)
        n = min(args.num_train, len(base))
        ds = Corrupted(base, n, args.corrupt or 0.0, args.seed)
        train_loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
        test = vc.vpg.small_dataset(args.dataset, str(vc.ROOT / "data"), train=False)
        layer = vc.last_linear(model)
        print(f"model : {tag}  (VeLPIC from {args.corrupt:.0%}-corrupted labels)")
    else:
        spec = args.model or (ROBUST[args.dataset] if args.robust else STANDARD[args.dataset])
        tag = f"{args.dataset}_{vc.vpg.safe(spec)}"
        model, layer, train_loader, test_loader, _, _ = vc.load_setup(
            args.dataset, spec, device, args.batch_size, args.workers, args.num_train)
        test = test_loader.dataset
        print(f"model : {spec}")

    X, y = vc.penultimate(model, layer, train_loader, device)
    W_velpic, _, _ = vc.velpic_vectors(X, y, 10, verbose=False)
    W_orig = layer.weight.detach().float().cpu().clone()
    heads = vc.build_heads(HEADS, W_orig, W_velpic, X, y, 10)

    loader = DataLoader(Subset(test, range(min(args.num_eval, len(test)))),
                        batch_size=args.batch_size, num_workers=args.workers)
    total = min(args.num_eval, len(test))

    print("\n  head          clean     PGD(white-box)   PGD(transfer from orig)")
    res = {}
    for name, W in heads.items():
        vc.set_head(layer, W)
        clean = adv = 0
        for x, yt in loader:
            x, yt = x.to(device), yt.to(device)
            with torch.no_grad():
                clean += (model(x).argmax(1) == yt).sum().item()
            xa = pgd(model, x, yt, args.eps, args.alpha, args.steps)
            with torch.no_grad():
                adv += (model(xa).argmax(1) == yt).sum().item()
        res[name] = {"clean": 100 * clean / total, "wb": 100 * adv / total}

    vc.set_head(layer, W_orig)
    advs = []
    for x, yt in loader:
        x, yt = x.to(device), yt.to(device)
        advs.append((pgd(model, x, yt, args.eps, args.alpha, args.steps).cpu(), yt.cpu()))
    for name, W in heads.items():
        vc.set_head(layer, W)
        c = 0
        with torch.no_grad():
            for xa, yt in advs:
                c += (model(xa.to(device)).argmax(1) == yt.to(device)).sum().item()
        res[name]["tr"] = 100 * c / total
    vc.set_head(layer, W_orig)

    for name, r in res.items():
        print(f"  {name:<12} {r['clean']:6.2f}%      {r['wb']:6.2f}%          {r['tr']:6.2f}%")

    path = out / f"pgd_{tag}.csv"
    with open(path, "w") as f:
        f.write("head,clean,pgd_whitebox,pgd_transfer_from_orig\n")
        for name, r in res.items():
            f.write(f"{name},{r['clean']},{r['wb']},{r['tr']}\n")
    print(f"\nsaved: {path}")


if __name__ == "__main__":
    main()
