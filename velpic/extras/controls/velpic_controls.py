"""
velpic_controls.py  --  the same experiment plus the controls that make it
interpretable. Everything here is beyond what was asked for.

main/ compares three heads: orig, velpic, resid. That comparison alone cannot
tell you whether a result means anything, because it has no reference point.
This script adds three things:

  rand        random unit vectors in the head. If this produces a similar
              gradient map to the real head, then the head is not what decides
              the map. It does, which is the central finding.
  classmean   the normalised class means. Expected to match velpic exactly,
              because VeLPIC on a post-ReLU layer IS nearest-class-mean.
  resid_orth  w_j - (w_j . v_j) v_j, the scale-free residual. Plain subtraction
              is ill-posed: it gave 94.76% on one model and 0.10% on another
              purely from the ratio of vector lengths.

It also runs the adversarially robust counterpart model, which is the control
that proves the gradient method can detect robustness at all:

  cifar10   Standard      vs  Carmon2019Unlabeled
  imagenet  Standard_R50  vs  Salman2020Do_R50

and reports the participation ratio, an objective stand-in for "does this
attribution look like the object" -- low means concentrated and structured,
high means diffuse noise.

  python velpic_controls.py --dataset cifar10
  python velpic_controls.py --dataset cifar10 --model robustbench:cifar10:Carmon2019Unlabeled
  python velpic_controls.py --dataset imagenet --robust
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "lib"))
import velpic_core as vc                                  # noqa: E402

STANDARD = {
    "mnist": "smallcnn",
    "cifar10": "robustbench:cifar10:Standard",
    "imagenet": "robustbench:imagenet:Standard_R50",
}
ROBUST = {
    "cifar10": "robustbench:cifar10:Carmon2019Unlabeled",
    "imagenet": "robustbench:imagenet:Salman2020Do_R50",
}
HEADS = ["orig", "velpic", "resid", "resid_orth", "rand", "classmean"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["mnist", "cifar10", "imagenet"])
    p.add_argument("--model", default=None)
    p.add_argument("--robust", action="store_true",
                   help="use the adversarially trained counterpart model")
    p.add_argument("--num_train", type=int, default=10000)
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--score", choices=["logit", "softmax"], default="logit")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--skip_acc", action="store_true")
    p.add_argument("--out", default=str(HERE / "outputs"))
    args = p.parse_args()

    spec = args.model or (ROBUST[args.dataset] if args.robust else STANDARD[args.dataset])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    _, _, _, vwc = vc.import_repo()
    device = vwc.dev
    print(f"device : {device}\ndataset: {args.dataset}\nmodel  : {spec}")

    model, layer, train_loader, test_loader, names, M = vc.load_setup(
        args.dataset, spec, device, args.batch_size, args.workers, args.num_train)
    print(f"head   : ({M}, {layer.in_features})")

    X, y = vc.penultimate(model, layer, train_loader, device)
    print(f"\n[1] X = {tuple(X.shape)}   min = {float(X.min()):.3f}")

    print("\n[2] VeLPIC class vectors")
    W_velpic, _, _ = vc.velpic_vectors(X, y, M)
    W_orig = layer.weight.detach().float().cpu().clone()
    heads = vc.build_heads(HEADS, W_orig, W_velpic, X, y, M)
    print(f"\n    orig row norm {float(W_orig.norm(dim=1).mean()):.3f}   "
          f"velpic row norm {float(W_velpic.norm(dim=1).mean()):.3f}   "
          f"cos {float(torch.nn.functional.cosine_similarity(W_orig, W_velpic).mean()):+.4f}")

    accs = {}
    if not args.skip_acc:
        print("\n[3] test accuracy per head")
        for name, W in heads.items():
            vc.set_head(layer, W)
            accs[name] = vc.accuracy(model, test_loader, device)
            print(f"      {name:<12} {accs[name]:6.2f}%")
        vc.set_head(layer, W_orig)

    print(f"\n[4] gradient maps for {args.num_images} test images")
    picks = vc.sample_images(args.dataset, args.num_images)
    rows, prs = vc.gradient_rows(model, layer, heads, W_orig, picks, names,
                                 device, args.score)
    print("\n    participation ratio (LOW = structured, HIGH = diffuse noise)")
    for n, v in prs.items():
        print(f"      {n:<12} {v:9.1f}")

    tag = f"{args.dataset}_{vc.vpg.safe(spec)}"
    path = vc.draw(rows, f"{args.dataset} -- {spec}  (with controls)",
                   out / f"controls_{tag}.png")
    with open(out / f"results_{tag}.csv", "w") as f:
        f.write("head,test_acc,participation_ratio\n")
        for n in heads:
            f.write(f"{n},{accs.get(n,'')},{prs[n]}\n")
    print(f"\nDONE\nsaved: {path}")


if __name__ == "__main__":
    main()
