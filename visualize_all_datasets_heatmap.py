
import argparse
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


# ============================================================
# ONE SCRIPT FOR GRADIENT VISUALIZATION
#
# Datasets:
#   mnist
#   fashion_mnist
#   cifar10
#   gtsrb
#   imagenet
#   imagefolder
#
# Models:
#   For ImageNet: pretrained torchvision models
#   For MNIST/Fashion/CIFAR/GTSRB: automatically trains a small
#   CNN the first time, saves it, and reuses it afterwards.
#   For CIFAR-10 robust model: RobustBench option.
#
# Usage examples are given at the bottom of this file.
# ============================================================


MNIST_MEAN, MNIST_STD = [0.1307], [0.3081]
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD  = [0.2470, 0.2435, 0.2616]
IM_MEAN = [0.485, 0.456, 0.406]
IM_STD  = [0.229, 0.224, 0.225]


class SmallCNN(nn.Module):
    def __init__(self, in_channels, num_classes, image_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FlatImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        self.files = sorted([
            p for p in Path(root).iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ])
        if not self.files:
            raise FileNotFoundError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img), -1


def args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        required=True,
        choices=["mnist", "fashion_mnist", "cifar10", "gtsrb", "imagenet", "imagefolder"]
    )
    p.add_argument("--data_root", default="./data")
    p.add_argument("--num_images", type=int, default=10)
    p.add_argument("--epochs", type=int, default=3,
                   help="Only used the first time a small model must be trained.")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--model", default="auto",
                   help="auto, resnet18, resnet50, squeezenet1_1, robustbench:MODEL")
    p.add_argument("--target", choices=["true", "pred"], default="true")
    p.add_argument("--output_dir", default="gradient_outputs")
    return p.parse_args()


def make_data(name, root):
    name = name.lower()

    if name == "mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(MNIST_MEAN, MNIST_STD)])
        ds = datasets.MNIST(root, train=False, download=True, transform=tf)
        return ds, 10, 1, (28, 28)

    if name == "fashion_mnist":
        tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(MNIST_MEAN, MNIST_STD)])
        ds = datasets.FashionMNIST(root, train=False, download=True, transform=tf)
        return ds, 10, 1, (28, 28)

    if name == "cifar10":
        tf = transforms.Compose([transforms.ToTensor(),
                                  transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        ds = datasets.CIFAR10(root, train=False, download=True, transform=tf)
        return ds, 10, 3, (32, 32)

    if name == "gtsrb":
        # torchvision expects the dataset root containing GTSRB
        tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
        ds = datasets.GTSRB(root, split="test", download=True, transform=tf)
        return ds, 43, 3, (32, 32)

    if name in ("imagenet", "imagefolder"):
        tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IM_MEAN, IM_STD)
        ])

        rootp = Path(root)

        # Flat folder of images = labels unavailable, so we use model prediction.
        flat = [
            x for x in rootp.iterdir()
            if x.is_file() and x.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".JPEG".lower()}
        ] if rootp.exists() else []

        if flat:
            return FlatImageDataset(root, tf), 1000, 3, (224, 224)

        # ImageFolder structure: root/class/image.jpg
        if rootp.exists():
            return datasets.ImageFolder(root, transform=tf), 1000, 3, (224, 224)

        raise FileNotFoundError(f"Image folder not found: {root}")

    raise ValueError(name)


def build_small_model(in_channels, num_classes):
    return SmallCNN(in_channels, num_classes, 32)


def train_small_model(dataset_name, model, train_ds, device, epochs, batch_size, checkpoint):
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    print(f"Training {dataset_name} model for {epochs} epoch(s)...")

    for ep in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()

            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.numel()

        print(
            f"epoch {ep+1}/{epochs} "
            f"loss={total_loss/total:.4f} "
            f"acc={100*correct/total:.2f}%"
        )

    torch.save(model.state_dict(), checkpoint)
    print(f"Saved model -> {checkpoint}")
    model.eval()


def load_or_train_small(dataset_name, root, in_channels, num_classes, device, epochs, batch_size):
    checkpoint_dir = Path(".") / "saved_models"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint = checkpoint_dir / f"{dataset_name}_smallcnn.pth"

    model = build_small_model(in_channels, num_classes)

    if checkpoint.exists():
        print(f"Loading cached model: {checkpoint}")
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        model.to(device).eval()
        return model

    # Load training split only for the datasets that support automatic download.
    if dataset_name == "mnist":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
        train_ds = datasets.MNIST(root, train=True, download=True, transform=tf)

    elif dataset_name == "fashion_mnist":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD)
        ])
        train_ds = datasets.FashionMNIST(root, train=True, download=True, transform=tf)

    elif dataset_name == "cifar10":
        tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
        train_ds = datasets.CIFAR10(root, train=True, download=True, transform=tf)

    elif dataset_name == "gtsrb":
        tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])
        train_ds = datasets.GTSRB(root, split="train", download=True, transform=tf)

    else:
        raise ValueError(f"Automatic training not configured for {dataset_name}")

    # Cap the first-run training set so the first run stays manageable.
    max_train = {"mnist": 20000, "fashion_mnist": 20000,
                 "cifar10": 30000, "gtsrb": 30000}[dataset_name]

    if len(train_ds) > max_train:
        train_ds = Subset(train_ds, range(max_train))

    train_small_model(
        dataset_name, model, train_ds, device, epochs, batch_size, checkpoint
    )

    return model


def load_model(dataset_name, model_name, data_root, in_channels, num_classes,
               device, epochs, batch_size):

    if model_name.startswith("robustbench:"):
        try:
            from robustbench.utils import load_model as rb_load_model
        except Exception as e:
            raise RuntimeError(
                'Run: pip install "setuptools<81"\n'
                "then: pip install git+https://github.com/RobustBench/robustbench.git"
            ) from e

        parts = model_name.split(":")
        if len(parts) != 3:
            raise ValueError(
                "Use --model robustbench:cifar10:MODEL_NAME"
            )

        _, dataset, rb_model = parts
        model = rb_load_model(
            model_name=rb_model,
            dataset=dataset,
            threat_model="Linf"
        )
        return model.to(device).eval()

    if dataset_name in {"mnist", "fashion_mnist", "cifar10", "gtsrb"}:
        return load_or_train_small(
            dataset_name, data_root, in_channels, num_classes,
            device, epochs, batch_size
        )

    # ImageNet pretrained torchvision models.
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif model_name == "resnet50" or model_name == "auto":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif model_name == "squeezenet1_1":
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    elif model_name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(device).eval()


def denormalize(x, dataset_name):
    x = x.detach().cpu().clone()

    if dataset_name in {"mnist", "fashion_mnist"}:
        mean = torch.tensor(MNIST_MEAN)[:, None, None]
        std = torch.tensor(MNIST_STD)[:, None, None]
    elif dataset_name == "cifar10" or dataset_name == "gtsrb":
        mean = torch.tensor(CIFAR_MEAN)[:, None, None]
        std = torch.tensor(CIFAR_STD)[:, None, None]
    else:
        mean = torch.tensor(IM_MEAN)[:, None, None]
        std = torch.tensor(IM_STD)[:, None, None]

    if x.shape[0] == 1 and mean.shape[0] == 3:
        mean = mean[:1]
        std = std[:1]

    x = x * std + mean
    return torch.clamp(x, 0, 1)


def grad_visual(grad):
    # Heatmap of gradient magnitude, similar to the paper-style
    # red/black visualizations: black = low importance, hot = high.
    g = grad.detach().cpu()

    if g.shape[0] > 1:
        # Collapse RGB channels to one pixel-level magnitude.
        z = g.abs().mean(dim=0)
    else:
        z = g.abs()[0]

    # Clip extreme values so the heatmap is not dominated by a few pixels.
    mu = z.mean()
    sd = z.std() + 1e-8
    z = torch.clamp(z, mu - 3 * sd, mu + 3 * sd)

    # Rescale to [0, 1].
    z = (z - z.min()) / (z.max() - z.min() + 1e-8)

    return z.numpy()

def label_name(ds, y):
    if y < 0:
        return "N/A"
    if hasattr(ds, "classes") and y < len(ds.classes):
        return str(ds.classes[y])
    return str(y)


def main():
    a = args()
    outdir = Path(a.output_dir)
    outdir.mkdir(exist_ok=True)

    # Keep this folder PNG-only: remove old raw-gradient files from
    # previous versions of the script.
    for old_npy in outdir.glob("*.npy"):
        try:
            old_npy.unlink()
        except OSError:
            pass

    torch.manual_seed(42)
    np.random.seed(42)

    print(f"device: {a.device}")

    ds, num_classes, in_channels, _ = make_data(a.dataset, a.data_root)

    if len(ds) < a.num_images:
        n = len(ds)
    else:
        n = a.num_images

    model = load_model(
        a.dataset, a.model, a.data_root,
        in_channels, num_classes,
        a.device, a.epochs, a.batch_size
    )

    criterion = nn.CrossEntropyLoss()
    rows = []

    for i in range(n):
        x, y = ds[i]
        y_int = int(y) if torch.is_tensor(y) else int(y)

        x = x.unsqueeze(0).to(a.device)
        x.requires_grad_(True)

        logits = model(x)
        pred = int(logits.argmax(1).item())

        if a.target == "true" and y_int >= 0:
            target = y_int
        else:
            target = pred

        loss = criterion(logits, torch.tensor([target], device=a.device))

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()

        grad = x.grad[0].detach().cpu()

        original = denormalize(x[0], a.dataset)
        original_np = np.transpose(original.numpy(), (1, 2, 0))
        if original_np.shape[2] == 1:
            original_np = original_np[:, :, 0]

        gnp = grad_visual(grad)

        true_s = label_name(ds, y_int)
        pred_s = label_name(ds, pred)

        print(
            f"{i+1:02d}/{n}  "
            f"true={true_s}  pred={pred_s}  loss={loss.item():.5f}"
        )

        rows.append((original_np, gnp, true_s, pred_s))

    fig, axes = plt.subplots(
        n, 2,
        figsize=(8, max(2.3 * n, 5))
    )

    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (img, g, true_s, pred_s) in enumerate(rows):
        axes[i, 0].imshow(img, cmap="gray" if img.ndim == 2 else None)
        axes[i, 0].set_title(f"Original | true={true_s} pred={pred_s}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(g, cmap="hot", vmin=0, vmax=1)
        axes[i, 1].set_title("Loss gradient heatmap")
        axes[i, 1].axis("off")

    plt.tight_layout()

    filename = outdir / f"{a.dataset}_{a.model.replace(':','_')}_10.png"
    plt.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close()

    print("\nDONE")
    print(f"figure: {filename}")
    print(f"raw gradients: {outdir}")


if __name__ == "__main__":
    main()
