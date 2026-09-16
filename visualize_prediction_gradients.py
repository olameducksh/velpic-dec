"""
visualize_prediction_gradients.py

Plain "gradients at the image" visualization from Section 2.1 of
    Sundararajan, Taly, Yan - "Gradients of Counterfactuals" (arXiv:1611.02639)
(NOT integrated / interior gradients).

For every image:
  1. Forward pass -> softmax probabilities. L = highest-scoring (predicted) class.
  2. Score F_L(x) = softmax probability of class L.
  3. Gradient of F_L(x) w.r.t. the input image pixels (pixel values in [0, 1]).
  4. Pixel importance  P[i,j] = sum over c in {R,G,B} of |dF_L / dx[i,j,c]|   (Eq. 1)
  5. Visualization: scale each pixel of the ORIGINAL image by P[i,j]
     (brighter = more important, dark = unimportant).

Output: <output_dir>/<dataset>_<model>_prediction_grad.png
  one row per image, columns = original | pixel importance P | image x P

Examples (run from the project folder, one model per run):
  python visualize_prediction_gradients.py --dataset mnist --num_images 10
  python visualize_prediction_gradients.py --dataset fashion_mnist --num_images 10
  python visualize_prediction_gradients.py --dataset cifar10 --num_images 10
  python visualize_prediction_gradients.py --dataset cifar10 --model robustbench:cifar10:Carmon2019Unlabeled --num_images 10
  python visualize_prediction_gradients.py --dataset imagenet --data_root .\\imagenet_samples --model resnet50 --num_images 10
  python visualize_prediction_gradients.py --dataset imagenet --data_root .\\imagenet_samples --model squeezenet1_1 --num_images 10
"""

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


MNIST_MEAN, MNIST_STD = [0.1307], [0.3081]
CIFAR_MEAN, CIFAR_STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

SMALL_DATASETS = {"mnist", "fashion_mnist", "cifar10"}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
class SmallCNN(nn.Module):
    """Same architecture as in the earlier scripts, so saved_models/*.pth load."""

    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class WithNormalization(nn.Module):
    """Takes images in [0, 1] and normalizes them inside the model.

    This way the gradient is taken w.r.t. the real pixel values, as in the paper,
    and every model (normalized or not) is fed the same kind of input.
    """

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))

    def forward(self, x):
        return self.model((x - self.mean) / self.std)


def small_dataset(name, root, train):
    tf = transforms.ToTensor()  # pixels in [0, 1]; normalization happens in the model
    if name == "mnist":
        return datasets.MNIST(root, train=train, download=True, transform=tf)
    if name == "fashion_mnist":
        return datasets.FashionMNIST(root, train=train, download=True, transform=tf)
    if name == "cifar10":
        return datasets.CIFAR10(root, train=train, download=True, transform=tf)
    raise ValueError(name)


def load_smallcnn(dataset, data_root, device, epochs, batch_size):
    in_ch = 1 if dataset in {"mnist", "fashion_mnist"} else 3
    mean, std = (MNIST_MEAN, MNIST_STD) if in_ch == 1 else (CIFAR_MEAN, CIFAR_STD)
    model = WithNormalization(SmallCNN(in_ch, 10), mean, std)

    ckpt = Path("saved_models") / f"{dataset}_smallcnn.pth"
    if ckpt.exists():
        print(f"  loading {ckpt}")
        model.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        return model.to(device).eval()

    # No checkpoint yet: train a small model once (same recipe as before).
    print(f"  {ckpt} not found -> training for {epochs} epoch(s)")
    ds = small_dataset(dataset, data_root, train=True)
    ds = Subset(ds, range(min(len(ds), 20000 if in_ch == 1 else 30000)))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        correct = total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            nn.functional.cross_entropy(out, y).backward()
            opt.step()
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()
        print(f"  epoch {ep + 1}/{epochs} train acc={100 * correct / total:.2f}%")
    ckpt.parent.mkdir(exist_ok=True)
    torch.save(model.model.state_dict(), ckpt)
    print(f"  saved {ckpt}")
    return model.eval()


def load_robustbench(spec, dataset, device, model_dir):
    # accepts robustbench:NAME or robustbench:DATASET:NAME
    parts = spec.split(":")
    rb_dataset, rb_name = (dataset, parts[1]) if len(parts) == 2 else (parts[1], parts[2])
    try:
        from robustbench.utils import load_model as rb_load_model
    except Exception as e:
        raise RuntimeError(
            'RobustBench not installed. Run: pip install "setuptools<81" then '
            "pip install git+https://github.com/RobustBench/robustbench.git"
        ) from e
    # RobustBench models expect raw [0, 1] images (normalization is built in).
    model = rb_load_model(model_name=rb_name, dataset=rb_dataset,
                          threat_model="Linf", model_dir=model_dir)
    return model.to(device).eval()


def load_model(spec, args):
    """Returns (model taking [0,1] images, (resize, crop) or None)."""
    if spec.startswith("robustbench:"):
        size = (256, 224) if args.dataset == "imagenet" else None
        return load_robustbench(spec, args.dataset, args.device, args.rb_model_dir), size

    if spec == "smallcnn":
        if args.dataset not in SMALL_DATASETS:
            raise ValueError("smallcnn is only for mnist / fashion_mnist / cifar10")
        return load_smallcnn(args.dataset, args.data_root, args.device,
                             args.epochs, args.batch_size), None

    if args.dataset != "imagenet":
        raise ValueError(f"'{spec}' is an ImageNet model; use smallcnn or robustbench:... "
                         f"for {args.dataset}")
    # Any torchvision ImageNet classifier: resnet18, resnet50, vgg16, densenet121,
    # squeezenet1_1, mobilenet_v3_small, efficientnet_b0, inception_v3, googlenet, ...
    weights = models.get_model_weights(spec).DEFAULT
    net = models.get_model(spec, weights=weights)
    tf = weights.transforms()
    size = (tf.resize_size[0], tf.crop_size[0])
    return WithNormalization(net, IM_MEAN, IM_STD).to(args.device).eval(), size


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_images(args):
    """Returns a list of dicts: {'data': tensor-in-[0,1] or PIL, 'true': str|None, 'name': str}
    plus the list of class names."""
    items = []
    if args.dataset in SMALL_DATASETS:
        ds = small_dataset(args.dataset, args.data_root, train=False)
        for i in range(args.start_index, min(args.start_index + args.num_images, len(ds))):
            x, y = ds[i]
            items.append({"data": x, "true": ds.classes[int(y)], "name": f"test{i}"})
        return items, list(ds.classes)

    classes = models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"]
    root = Path(args.data_root)
    if not root.exists():
        raise FileNotFoundError(f"Image folder not found: {root}")
    files = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    labeled = []
    if files:  # flat folder; true label guessed from names like n01440764_tench.JPEG
        for p in files:
            m = re.match(r"n\d{8}_(.+)", p.stem)
            labeled.append((p, m.group(1).replace("_", " ") if m else None))
    else:      # ImageFolder layout: root/<class>/<image>
        for d in sorted(x for x in root.iterdir() if x.is_dir()):
            labeled += [(p, d.name) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]
    if not labeled:
        raise FileNotFoundError(f"No images found in {root}")
    for p, true in labeled[args.start_index:args.start_index + args.num_images]:
        items.append({"data": Image.open(p).convert("RGB"), "true": true, "name": p.stem})
    return items, classes


def to_input(item, size):
    """Model-specific preprocessing -> 1xCxHxW tensor in [0, 1]."""
    if size is None:
        return item["data"].unsqueeze(0)
    resize, crop = size
    tf = transforms.Compose([transforms.Resize(resize), transforms.CenterCrop(crop),
                             transforms.ToTensor()])
    return tf(item["data"]).unsqueeze(0)


# ----------------------------------------------------------------------------
# The method
# ----------------------------------------------------------------------------
def prediction_gradient(model, x, score_type="softmax"):
    """Steps 1-4: returns (pred class, softmax prob, gradient CxHxW, importance HxW)."""
    x = x.clone().requires_grad_(True)
    with torch.enable_grad():
        logits = model(x)
        # softmax in float64: for very confident predictions (p ~ 1) float32 would
        # round p to exactly 1 and the gradient would become exactly 0.
        probs = torch.softmax(logits.double(), dim=1)
        pred = int(probs.argmax(dim=1).item())
        score = probs[0, pred] if score_type == "softmax" else logits[0, pred]
        (grad,) = torch.autograd.grad(score, x)

    grad = grad[0].detach().float().cpu()          # C x H x W
    importance = grad.abs().sum(dim=0).numpy()      # Eq. 1: sum_c |grad|
    return pred, probs[0, pred].item(), grad, importance


def normalize_importance(importance, percentile):
    """Map P to [0, 1] for display. Gradients are very spiky, so dividing by the max
    makes almost everything black; we divide by a high percentile and clip instead
    (use --clip_percentile 100 to divide by the max)."""
    hi = np.percentile(importance, percentile)
    if hi <= 0:
        hi = importance.max()
    if hi <= 0:
        return np.zeros_like(importance)
    return np.clip(importance / hi, 0.0, 1.0)


def make_overlay(image, p_norm):
    """Step 5: scale the original pixel intensities by the importance."""
    if image.ndim == 2:
        return image * p_norm
    return image * p_norm[..., None]


def to_numpy_image(x):
    img = x[0].detach().cpu().numpy()
    return img[0] if img.shape[0] == 1 else np.transpose(img, (1, 2, 0))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["mnist", "fashion_mnist", "cifar10", "imagenet"])
    p.add_argument("--model", default="auto",
                   help="auto (smallcnn for mnist/fashion_mnist/cifar10, resnet50 for imagenet) | "
                        "smallcnn | robustbench:cifar10:NAME | any torchvision ImageNet model "
                        "(resnet18, resnet50, squeezenet1_1, inception_v3, vgg16, ...)")
    p.add_argument("--data_root", default=None,
                   help="./data for mnist/fashion_mnist/cifar10, ./imagenet_samples for imagenet")
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--target", choices=["pred"], default="pred",
                   help="Paper method: always the highest-scoring (predicted) class.")
    p.add_argument("--score", choices=["softmax", "logit"], default="softmax",
                   help="What to differentiate. The paper uses the softmax score.")
    p.add_argument("--clip_percentile", type=float, default=99.0,
                   help="Display only: importance is divided by this percentile and clipped to 1.")
    p.add_argument("--output_dir", default="prediction_gradient_outputs")
    p.add_argument("--save_individual", action="store_true",
                   help="Also save each overlay as its own PNG.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rb_model_dir", default="./models", help="Where RobustBench stores checkpoints.")
    p.add_argument("--epochs", type=int, default=3, help="Only if a smallcnn has to be trained.")
    p.add_argument("--batch_size", type=int, default=128)
    a = p.parse_args()
    if a.model == "auto":
        a.model = "resnet50" if a.dataset == "imagenet" else "smallcnn"
    a.data_root = a.data_root or ("./imagenet_samples" if a.dataset == "imagenet" else "./data")
    return a


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def main():
    args = get_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"device: {args.device} | dataset: {args.dataset} | model: {args.model}")

    items, classes = load_images(args)
    n = len(items)
    model, size = load_model(args.model, args)
    tag = f"{args.dataset}_{safe(args.model)}"

    rows, correct, labeled = [], 0, 0
    for k, item in enumerate(items):
        x = to_input(item, size).to(args.device)
        pred, prob, grad, importance = prediction_gradient(model, x, args.score)
        img = to_numpy_image(x)
        p_norm = normalize_importance(importance, args.clip_percentile)
        overlay = make_overlay(img, p_norm)

        pred_name = classes[pred] if pred < len(classes) else str(pred)
        if item["true"] is not None:
            labeled += 1
            correct += int(item["true"].lower() == pred_name.lower())
        print(f"{k + 1:02d}/{n} {item['name']:<28} true={item['true']} "
              f"pred={pred_name} p={prob:.4f} max|grad|={grad.abs().max().item():.2e}")
        rows.append((img, p_norm, overlay, item["true"], f"{pred_name} ({prob:.2f})"))

        if args.save_individual:
            d = out / tag
            d.mkdir(exist_ok=True)
            plt.imsave(d / f"{k:02d}_{safe(item['name'])}_overlay.png", overlay,
                       cmap="gray" if overlay.ndim == 2 else None, vmin=0, vmax=1)
    if labeled:
        print(f"accuracy on labeled images shown: {correct}/{labeled}")

    # Figure: original | pixel importance | overlay  (one row per image)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
    for r, (img, p_norm, overlay, true, pred_s) in enumerate(rows):
        axes[r, 0].imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1)
        axes[r, 0].set_title(f"true: {true}\npred: {pred_s}" if true else f"pred: {pred_s}", fontsize=9)
        axes[r, 1].imshow(p_norm, cmap="gray", vmin=0, vmax=1)
        axes[r, 1].set_title(f"pixel importance\nsum_c |d {args.score} / d x|", fontsize=9)
        axes[r, 2].imshow(overlay, cmap="gray" if overlay.ndim == 2 else None, vmin=0, vmax=1)
        axes[r, 2].set_title("image x importance", fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    # tight_layout leaves just enough room above the axes for the column titles, so
    # free another half inch on top of that for the figure title; otherwise on a tall
    # figure the suptitle lands on the first row's column titles.
    h = fig.get_figheight()
    fig.subplots_adjust(top=fig.subplotpars.top - 0.5 / h)
    fig.suptitle(f"Gradients at the image - {args.dataset} / {args.model}",
                 fontsize=12, y=1 - 0.1 / h, va="top")
    path = out / f"{tag}_prediction_grad.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nDONE\nsaved: {path}")


if __name__ == "__main__":
    main()
