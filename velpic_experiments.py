"""code to choose the fitting/evaluation images"""
import csv
import json

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset

import pgd
import velpic as vp


def main():
    p = pgd.parser()
    p.description = __doc__
    p.add_argument("--fit", choices=["clean", "pgd", "mixed"], default="clean") # which training imgs to construct velpic withh
    p.add_argument("--test", choices=["clean", "pgd", "both"], default="both") # which test images to use for grads
    p.add_argument("--head", choices=["original", "velpic", "resid", "all"], default="all") # which head to use in the nn (both swapped heads have zero bias)
    p.add_argument("--n_show", "--n-show", type=int, default=8)
    p.add_argument("--output-dir", type=pgd.Path, default=vp.OUT / "adversarial")
    a = p.parse_args()
    fitting = a.head != "original"
    needs_pgd = (fitting and a.fit != "clean") or a.test != "clean"

    model, tr, te = pgd.load(a.dataset)

    base = a.output_dir / a.dataset
    folder, attack_meta = pgd.location(a) if needs_pgd else (None, None)
    tag = folder.name if needs_pgd else "clean"
    fit = a.fit if fitting else "none"
    out = base / "runs" / tag / f"fit_{fit}_head_{a.head}_test_{a.test}"
    out.mkdir(parents=True, exist_ok=True)
    heads = {"orig": model} if a.head in ("original", "all") else {}

    if fitting:
        fit_dir = base / "fits" / ("clean" if fit == "clean" else f"{fit}_{tag}")
        fit_dir.mkdir(parents=True, exist_ok=True)
        weights = fit_dir / "V.pt"
        
        if weights.exists():
            V = torch.load(weights, map_location="cuda", weights_only=True)
        else:
            train_set = tr
            if fit != "clean":
                adv = pgd.dataset(model, tr, "train", folder, attack_meta, a.bs)
                train_set = adv if fit == "pgd" else ConcatDataset([tr, adv])

            feats, ys = vp.features(model, train_set, a.bs)
            V = vp.velpic(feats, ys, len(te.classes), fit_dir / "pca")
            torch.save(V.cpu(), weights)  # final sign-corrected V, unlike the intermediate PC files
            (fit_dir / "metadata.json").write_text(json.dumps(
                dict(dataset=a.dataset, fit=fit, images=len(train_set), attack=attack_meta if fit != "clean" else None), indent=2))
            del feats, ys, train_set
            if fit != "clean":
                del adv
        
        W = vp.head(model).weight.detach()
        if a.head in ("velpic", "all"):
            heads["velpic"] = vp.swap(model, V)
        if a.head in ("resid", "all"):
            heads["resid"] = vp.swap(model, W - V)

    # identical indices pair the clean/PGD figures, across heads and epsilon values.
    idx = torch.randperm(len(te), generator=torch.Generator().manual_seed(0))[:a.n_show].tolist()
    metadata = dict(dataset=a.dataset, fit=fit, head=a.head, test=a.test, attack=attack_meta,
                    train_images=len(tr), test_images=len(te), figure_indices=idx)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    with (out / "accuracy.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test", "head", "test_acc"])
        for split in ("clean", "pgd"):
            if a.test not in (split, "both"):
                continue
            test_set = te if split == "clean" else pgd.dataset(model, te, "test", folder, attack_meta, a.bs)
            with torch.no_grad():
                for name, m in heads.items():
                    acc = vp.vwc.accuracy_model(m, DataLoader(test_set, a.bs), f"{a.dataset} {fit} {split} {name}")
                    writer.writerow([split, name, f"{acc:.2f}"])
            x, y = next(iter(DataLoader(Subset(test_set, idx), len(idx))))
            vp.grads.plot(heads, x.cuda(), y, te.classes, out / f"grads_{split}.png",
                          f"{a.dataset}: fit={fit}, test={split}, epsilon={a.epsilon if needs_pgd else 'none'}")

if __name__ == "__main__":
    main()