import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms, models

MNIST_MEAN, MNIST_STD = [0.1307], [0.3081]
CIFAR_MEAN, CIFAR_STD = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
IM_MEAN, IM_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class SmallCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FlatImageDataset(Dataset):
    def __init__(self, root, transform):
        root = Path(root)
        self.files = sorted([p for p in root.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}])
        if not self.files:
            raise FileNotFoundError(f"No images found in {root}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        return self.transform(img), -1


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True,
                   choices=['mnist', 'fashion_mnist', 'cifar10', 'gtsrb', 'imagenet', 'imagefolder'])
    p.add_argument('--data_root', default='./data')
    p.add_argument('--num_images', type=int, default=10)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--model', default='auto')
    p.add_argument('--target', choices=['pred'], default='pred',
                    help='Paper method: always attribute the highest-scoring predicted class.')
    p.add_argument('--output_dir', default='prediction_gradient_outputs')
    return p.parse_args()


def make_data(name, root):
    name = name.lower()
    if name == 'mnist':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])
        return datasets.MNIST(root, train=False, download=True, transform=tf), 10, 1
    if name == 'fashion_mnist':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])
        return datasets.FashionMNIST(root, train=False, download=True, transform=tf), 10, 1
    if name == 'cifar10':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        return datasets.CIFAR10(root, train=False, download=True, transform=tf), 10, 3
    if name == 'gtsrb':
        tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        return datasets.GTSRB(root, split='test', download=True, transform=tf), 43, 3
    if name in ('imagenet', 'imagefolder'):
        tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(IM_MEAN, IM_STD)])
        rootp = Path(root)
        if not rootp.exists():
            raise FileNotFoundError(f'Image folder not found: {root}')
        flat = [p for p in rootp.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}]
        if flat:
            return FlatImageDataset(root, tf), 1000, 3
        return datasets.ImageFolder(root, transform=tf), 1000, 3
    raise ValueError(name)


def build_model(in_channels, num_classes):
    return SmallCNN(in_channels, num_classes)


def train_small(name, model, root, device, epochs, batch_size, in_channels):
    if name == 'mnist':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])
        ds = datasets.MNIST(root, train=True, download=True, transform=tf)
        max_train = 20000
    elif name == 'fashion_mnist':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MNIST_MEAN, MNIST_STD)])
        ds = datasets.FashionMNIST(root, train=True, download=True, transform=tf)
        max_train = 20000
    elif name == 'cifar10':
        tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        ds = datasets.CIFAR10(root, train=True, download=True, transform=tf)
        max_train = 30000
    elif name == 'gtsrb':
        tf = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])
        ds = datasets.GTSRB(root, split='train', download=True, transform=tf)
        max_train = 30000
    else:
        raise ValueError(name)

    if len(ds) > max_train:
        ds = Subset(ds, range(max_train))

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    model.to(device)

    print(f'Training {name} model for {epochs} epoch(s)...')
    model.train()
    for ep in range(epochs):
        total, correct, total_loss = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            total += y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total_loss += loss.item() * y.size(0)
        print(f'epoch {ep+1}/{epochs} loss={total_loss/total:.4f} acc={100*correct/total:.2f}%')
    return model.eval()


def load_model(args, in_channels, num_classes):
    if args.model.startswith('robustbench:'):
        try:
            from robustbench.utils import load_model as rb_load_model
        except Exception as e:
            raise RuntimeError(
                'RobustBench import failed. Run: pip install "setuptools<81" then '
                'pip install git+https://github.com/RobustBench/robustbench.git'
            ) from e
        _, ds_name, rb_name = args.model.split(':', 2)
        return rb_load_model(model_name=rb_name, dataset=ds_name, threat_model='Linf').to(args.device).eval()

    if args.dataset in {'mnist','fashion_mnist','cifar10','gtsrb'}:
        ckpt_dir = Path('saved_models')
        ckpt_dir.mkdir(exist_ok=True)
        ckpt = ckpt_dir / f'{args.dataset}_smallcnn.pth'
        model = build_model(in_channels, num_classes)
        if ckpt.exists():
            print(f'Loading cached model: {ckpt}')
            model.load_state_dict(torch.load(ckpt, map_location='cpu'))
            return model.to(args.device).eval()
        model = train_small(args.dataset, model, args.data_root, args.device, args.epochs, args.batch_size, in_channels)
        torch.save(model.state_dict(), ckpt)
        print(f'Saved model -> {ckpt}')
        return model

    if args.model in {'auto','resnet50'}:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    elif args.model == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif args.model == 'squeezenet1_1':
        model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT)
    elif args.model == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    else:
        raise ValueError(f'Unknown model: {args.model}')
    return model.to(args.device).eval()



def denormalize(x, dataset_name):
    x = x.detach().cpu()
    if dataset_name in {"mnist", "fashion_mnist"}:
        mean = torch.tensor(MNIST_MEAN).view(-1,1,1)
        std = torch.tensor(MNIST_STD).view(-1,1,1)
    elif dataset_name in {"cifar10", "gtsrb"}:
        mean = torch.tensor(CIFAR_MEAN).view(-1,1,1)
        std = torch.tensor(CIFAR_STD).view(-1,1,1)
    else:
        mean = torch.tensor(IM_MEAN).view(-1,1,1)
        std = torch.tensor(IM_STD).view(-1,1,1)
    if x.shape[0] == 1 and mean.shape[0] == 3:
        mean, std = mean[:1], std[:1]
    return torch.clamp(x * std + mean, 0, 1)


def original_np(x, dataset_name):
    img = denormalize(x, dataset_name).detach().cpu().numpy()
    if img.shape[0] == 1:
        return img[0]
    return np.transpose(img, (1,2,0))


def get_prediction(model, x):
    """Return the highest-scoring class and its softmax probability."""
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred = int(probs.argmax(dim=1).item())
    score = probs[:, pred]
    return pred, score


def get_target(model, x, y):
    # The paper always uses the highest-scoring predicted class.
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
    return pred


def make_overlay(image, importance):
    """Scale original pixel intensities in proportion to importance.

    This follows Eq. (1) + the visualization description in Section 2.1:
    higher-importance pixels remain brighter, low-importance pixels are
    darkened.
    """
    if image.ndim == 2:
        return np.clip(image * importance, 0, 1)

    return np.clip(image * importance[..., None], 0, 1)


def main():
    a = get_args()
    out = Path(a.output_dir)
    out.mkdir(exist_ok=True)

    # Keep the output folder PNG-only.
    for f in out.glob('*.npy'):
        try:
            f.unlink()
        except OSError:
            pass

    print('device:', a.device)
    ds, num_classes, in_channels = make_data(a.dataset, a.data_root)
    model = load_model(a, in_channels, num_classes)

    n = min(a.num_images, len(ds))
    rows = []

    for i in range(n):
        x, y = ds[i]
        y_int = int(y) if torch.is_tensor(y) else int(y)
        x = x.unsqueeze(0).to(a.device)
        x.requires_grad_(True)

        # ----------------------------------------------------
        # PAPER-STYLE PLAIN GRADIENT
        #
        # 1. Find the highest-scoring predicted class L.
        # 2. Take its SOFTMAX prediction score F_L(x).
        # 3. Differentiate F_L(x) with respect to the input image.
        # 4. Aggregate |gradient| over RGB channels (Eq. 1).
        # 5. Normalize the pixel scores for display.
        # 6. Scale original pixel intensities by those scores.
        # ----------------------------------------------------
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        prediction_score = probs[:, pred]

        model.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad.zero_()

        prediction_score.sum().backward()

        grad = x.grad.detach()[0]

        # P_ij = sum_c | dF_L / d x_ijc |
        importance = grad.abs()
        if importance.shape[0] > 1:
            importance = importance.sum(dim=0)
        else:
            importance = importance[0]

        importance = importance.cpu().numpy()

        # Visualization normalization only: map max importance to 1.
        importance = importance / (importance.max() + 1e-8)

        img = original_np(x[0], a.dataset)
        overlay = make_overlay(img, importance)

        true_name = (
            str(ds.classes[y_int])
            if hasattr(ds, 'classes') and y_int >= 0 and y_int < len(ds.classes)
            else ('N/A' if y_int < 0 else str(y_int))
        )
        pred_name = (
            str(ds.classes[pred])
            if hasattr(ds, 'classes') and pred < len(ds.classes)
            else str(pred)
        )

        print(
            f'{i+1:02d}/{n} '
            f'true={true_name} pred={pred_name} '
            f'score={prediction_score.item():.5f}'
        )

        rows.append((img, importance, overlay, true_name, pred_name))

    # Original | prediction-gradient map | overlay
    fig, axes = plt.subplots(
        n, 3,
        figsize=(11, max(2.6 * n, 6))
    )

    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, (img, importance, overlay, true_name, pred_name) in enumerate(rows):
        axes[i, 0].imshow(
            img,
            cmap='gray' if img.ndim == 2 else None
        )
        axes[i, 0].set_title(
            f'Original | true={true_name} pred={pred_name}'
        )
        axes[i, 0].axis('off')

        axes[i, 1].imshow(
            importance,
            cmap='hot',
            vmin=0,
            vmax=1
        )
        axes[i, 1].set_title(
            'Prediction-gradient importance'
        )
        axes[i, 1].axis('off')

        axes[i, 2].imshow(
            overlay,
            cmap='gray' if overlay.ndim == 2 else None
        )
        axes[i, 2].set_title(
            'Gradient overlay'
        )
        axes[i, 2].axis('off')

    plt.tight_layout()

    filename = out / (
        f'{a.dataset}_{a.model.replace(":", "_")}_prediction_grad.png'
    )
    plt.savefig(
        filename,
        dpi=180,
        bbox_inches='tight'
    )
    plt.close(fig)

    print('\nDONE')
    print('saved:', filename)


if __name__ == '__main__':
    main()
