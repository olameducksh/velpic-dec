"""
velpic_main.py  --  the experiment as specified, nothing else.

Three steps, three heads:

  1. build the VeLPIC class vectors from the penultimate layer
  2. replace the pre-softmax weights with them
  3. take the gradient of the prediction w.r.t. the input, and draw it

  orig    the trained weights, unchanged
  velpic  the VeLPIC vectors                      W_velpic
  resid   the vector subtraction                  W_orig - W_velpic

Everything beyond these three heads lives in ../extras.

The VeLPIC maths is the authors' own code, called unmodified. The gradient and
plotting code is visualize_prediction_gradients.py, also unmodified. The only
new lines are: a forward hook to read the penultimate layer, and
layer.weight.copy_(W) to swap the head.

  python velpic_main.py --dataset mnist
  python velpic_main.py --dataset cifar10
  python velpic_main.py --dataset imagenet
"""

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
import velpic_core as vc                                  # noqa: E402

DEFAULT_MODEL = {
    "mnist": "smallcnn",
    "fashion_mnist": "smallcnn",
    "cifar10": "robustbench:cifar10:Standard",
    "imagenet": "robustbench:imagenet:Standard_R50",
}
HEADS = ["orig", "velpic", "resid"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True,
                   choices=["mnist", "fashion_mnist", "cifar10", "imagenet"])
    p.add_argument("--model", default=None, help="defaults per dataset")
    p.add_argument("--num_train", type=int, default=10000,
                   help="training images used to build the vectors")
    p.add_argument("--num_images", type=int, default=8)
    p.add_argument("--score", choices=["logit", "softmax"], default="logit")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model_spec = args.model or DEFAULT_MODEL[args.dataset]
    out = Path(args.out) if args.out else HERE / "outputs" / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    _, _, _, vwc = vc.import_repo()
    device = vwc.dev
    print(f"device : {device}")
    print(f"dataset: {args.dataset}")
    print(f"model  : {model_spec}")

    model, layer, train_loader, test_loader, names, M = vc.load_setup(
        args.dataset, model_spec, device, args.batch_size, args.workers, args.num_train)
    d = layer.in_features
    print(f"head   : ({M}, {d})")

    # --- 1. penultimate activations ----------------------------------------
    print(f"\n[1] penultimate activations")
    X, y = vc.penultimate(model, layer, train_loader, device)
    print(f"    X = {tuple(X.shape)}   min = {float(X.min()):.3f}")

    # --- 2. VeLPIC vectors, and swap them in --------------------------------
    print("\n[2] VeLPIC class vectors")
    W_velpic, _, _ = vc.velpic_vectors(X, y, M)
    W_orig = layer.weight.detach().float().cpu().clone()
    heads = vc.build_heads(HEADS, W_orig, W_velpic, X, y, M)

    print(f"\n    orig row norm {float(W_orig.norm(dim=1).mean()):.3f}   "
          f"velpic row norm {float(W_velpic.norm(dim=1).mean()):.3f}   "
          f"cos(w_j, v_j) {float(torch.nn.functional.cosine_similarity(W_orig, W_velpic).mean()):+.4f}")

    print("\n    test accuracy per head")
    accs = {}
    for name, W in heads.items():
        vc.set_head(layer, W)
        accs[name] = vc.accuracy(model, test_loader, device)
        print(f"      {name:<8} {accs[name]:6.2f}%")
    vc.set_head(layer, W_orig)

    # --- 3. gradient maps ---------------------------------------------------
    print(f"\n[3] gradient maps for {args.num_images} test images")
    picks = vc.sample_images(args.dataset, args.num_images)
    rows, prs = vc.gradient_rows(model, layer, heads, W_orig, picks, names,
                                 device, args.score)
    for n, v in prs.items():
        print(f"      {n:<8} participation ratio {v:9.1f}")

    tag = f"{args.dataset}_{vc.vpg.safe(model_spec)}"
    path = vc.draw(rows, f"{args.dataset} -- {model_spec}", out / f"velpic_{tag}.png")

    with open(out / f"results_{tag}.csv", "w") as f:
        f.write("head,test_acc,participation_ratio\n")
        for n in heads:
            f.write(f"{n},{accs[n]},{prs[n]}\n")

    print(f"\nDONE\nsaved: {path}")


if __name__ == "__main__":
    main()
