"""pgd image generation"""
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import velpic as vp

sys.path.insert(0, str(vp.ROOT / "adversarial"))
from adv_attacks_main import attacks


class RawPixels(torch.nn.Module):
    def forward(self, x):
        return (x + 1) / 2


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=["mnist", "cifar10"])
    p.add_argument("--epsilon", type=float)
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--step-size", type=float, default=0.1)  # Normalized pixel units
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--cache-dir", type=Path)
    return p

def load(dataset):
    model, tr, te = vp.load(dataset, 0)
    model.requires_grad_(False)  # we need pixel gradients, not parameter gradients
    return model, tr, te


def location(a):
    meta = dict(dataset=a.dataset, epsilon=a.epsilon, steps=a.steps, step_size=a.step_size,
                batch_size=a.bs, targeted=False, random_start=False, norm="Linf")
    if a.dataset == "cifar10":
        meta.update(attack_input_range=[-1, 1], clip_range=[0, 1], checkpoint_input_range=[0, 1],
                    epsilon_coordinate="normalized")
    cache_dir = a.cache_dir or vp.HERE / "data" / ("pgd_mnist_cnn" if a.dataset == "mnist" else "pgd")
    folder = cache_dir / a.dataset / f"eps_{a.epsilon:g}"
    return folder, meta


def dataset(model, clean, split, folder, meta, bs):
    path = folder / f"{split}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        # print(f"Reusing {path} ({len(clean)} images)", flush=True)
        return TensorDataset(saved["images"], saved["labels"])
    params = {"eps": meta["epsilon"], "iters": meta["steps"],
              "step_size": meta["step_size"], "targeted": False}
    mnist = meta["dataset"] == "mnist"
    # CIFAR-10's checkpoint expects raw pixels, even while PGD works in normalized space.
    attack_model = model[1] if mnist else torch.nn.Sequential(RawPixels(), model)
    images, labels = [], []
    # print(f"Generating {split}: {len(clean)} images; epsilon={meta['epsilon']:g}", flush=True)
    for x, y in DataLoader(clean, bs): # x -> batch images, y -> true labels
        if params["eps"]:
            x_norm = model[0](x) if mnist else 2 * x - 1
            adv_norm = attacks.pgd_attack(attack_model, x_norm, y, params)
            adv = (adv_norm + 1) / 2  # Back to raw pixels for evaluation and plots.
        else:
            adv = x
        adv = adv.detach().cpu()  # save floating pixels, with no attack computation graph
        images.append(adv)
        labels.append(y)
    images, labels = torch.cat(images), torch.cat(labels) #joins all images into one tensor, all labels into another seperate tensor
    folder.mkdir(parents=True, exist_ok=True)
    
    # An interrupted write must not look like a completed dataset.
    torch.save({"images": images, "labels": labels, "metadata": meta}, path.with_suffix(".tmp"))
    path.with_suffix(".tmp").replace(path)

    (folder / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {path}", flush=True)
    return TensorDataset(images, labels)


def main():
    p = parser()
    a = p.parse_args()
    model, tr, te = load(a.dataset)
    folder, meta = location(a)
    for split, clean in {"train": tr, "test": te}.items():
        dataset(model, clean, split, folder, meta, a.bs)


if __name__ == "__main__":
    main()
