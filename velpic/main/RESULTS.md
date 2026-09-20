# VeLPIC weight swap — results

Just the experiment as specified: build the VeLPIC class vectors, replace the
last layer's weights with them, take the gradient of the prediction, draw it.
Three heads, three datasets.

| Head | What it is |
| --- | --- |
| `orig` | the trained weights, unchanged |
| `velpic` | the VeLPIC vectors, `W_velpic` |
| `resid` | the vector subtraction, `W_orig - W_velpic` |

Nothing else is in this folder. The controls that make these numbers
interpretable are in `../extras/controls`, and the reason they matter is at the
bottom of this page.

## Test accuracy

| Head | MNIST | CIFAR-10 | ImageNet (224px) |
| --- | --- | --- | --- |
| `orig` | 96.87% | 94.77% | 99.67% |
| `velpic` | **96.65%** | **94.76%** | **99.03%** |
| `resid` | **0.14%** | **94.76%** | **99.59%** |

**The weight swap works.** Replacing a trained final layer with ten
geometrically derived vectors costs between 0.01 and 1 percentage point. On
CIFAR-10 the two heads agree on 99.31% of individual test predictions. This
reproduces the paper's Section 8 claim on all three datasets.

**The subtraction does not define a usable second head.** Look along the
`resid` row: 0.14%, then 94.76%, then 99.59%. The same formula gives an
anti-classifier ten times worse than guessing on one dataset and a near-perfect
copy of the original on another.

The reason is that the two matrices are not on the same scale. VeLPIC vectors
always have length exactly 1, because the method normalises them. Trained
weight rows have whatever length training happened to give them:

| | `orig` row length | `velpic` row length | cos(w_j, v_j) | `resid` becomes |
| --- | --- | --- | --- | --- |
| MNIST | 0.630 | 1.000 | +0.339 | dominated by `-velpic` → 0.14% |
| CIFAR-10 | 3.035 | 1.000 | +0.709 | barely changed → 94.76% |
| ImageNet | 1.546 | 1.000 | +0.443 | barely changed → 99.59% |

When the trained rows are shorter than 1, subtracting flips the head into an
anti-classifier. When they are longer, subtracting is a rounding error. Neither
is "the non-robust part" — both are artefacts of an arbitrary length ratio.

A scale-free alternative, `w_j - (w_j . v_j) v_j`, is in
`../extras/controls`. It is well defined at any scale, and it scores 94.69% on
CIFAR-10 — so both halves of the split turn out to be fully competent
classifiers on their own.

## Gradient maps

The figures are in `outputs/{mnist,cifar10,imagenet}/`.

Read the `|grad|` columns, **not** the `overlay` columns. The overlay is the
original image multiplied by the importance map, so the object shows through
even when the map is pure noise. It is misleading for this comparison.

Participation ratio — how many pixels effectively carry the importance. Low
means concentrated and structured, high means diffuse noise:

| Head | MNIST (of 784) | CIFAR-10 (of 1024) | ImageNet (of 50176) |
| --- | --- | --- | --- |
| `orig` | 227.9 | 533.8 | 23273.2 |
| `velpic` | 211.7 | 533.2 | 24443.3 |
| `resid` | 206.9 | 532.7 | 23939.6 |

**The heads are indistinguishable.** Within each dataset every head lands within
a few percent of the others, and the maps look the same by eye. On CIFAR-10 the
three numbers agree to within 0.2%.

This holds even where the heads are doing completely different things. On MNIST
`orig` classifies at 96.87% and `resid` at 0.14% — one works and one is an
anti-classifier — yet their attribution maps score 227.9 and 206.9.

## What this means, and what is missing

Swapping the last layer changes what the network *predicts*. It does not change
what the network's attributions *look like*.

On its own, though, this comparison cannot tell you whether that is a real
finding or a limitation of the measurement. Three questions it cannot answer:

1. **Can this method detect a difference in attributions at all?** Maybe every
   model produces noisy maps and we would never see a difference.
2. **Is a similar map evidence of anything?** Maybe *any* matrix in the head
   gives a similar map, in which case similarity is meaningless.
3. **Is VeLPIC doing something specific?** Or would a simpler vector do the
   same?

`../extras/controls` answers all three, and the answers are what turn this from
an inconclusive result into a clear one:

1. Running an **adversarially robust** model through the identical pipeline
   produces dramatically different maps — object outlines instead of static,
   a participation ratio of 5,005 against 23,273 on ImageNet. The method works.
2. A **random** head still produces a map 73% correlated with the real one, and
   on a robust model traces the object just as cleanly. The head barely matters.
3. The **class-mean** head is identical to `velpic` (cosine 1.0000), because
   VeLPIC on a post-ReLU layer *is* a nearest-class-mean classifier.

Short version: attribution character is set by the feature extractor, not by
the last layer. See `../../VELPIC_REPORT.md` for the full argument and the
adversarial-robustness results.

## Reproducing

```
cd velpic/main
python -u velpic_main.py --dataset mnist
python -u velpic_main.py --dataset cifar10
python -u velpic_main.py --dataset imagenet
```

Models: SmallCNN for MNIST (checkpoint in `../../saved_models`), RobustBench
`Standard` (WideResNet-28-10) for CIFAR-10, RobustBench `Standard_R50`
(ResNet-50) for ImageNet. Vectors are built from 10,000 training images in each
case.

ImageNet here means **Imagenette**, a 10-class subset of real ImageNet at
224x224 — full 1000-class ImageNet needs 150 GB of training data. The
pre-softmax layer is sliced from 1000 to those 10 classes first, so all heads
are coherent 10-way classifiers.
