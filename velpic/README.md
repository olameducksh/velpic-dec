# VeLPIC experiments

Testing whether a network can be split into a "robust half" and a "non-robust
half" by editing only its last layer, using VeLPIC from Ketha & Ramaswamy,
[arXiv:2603.19865](https://arxiv.org/abs/2603.19865).

**Start with `main/`.** That folder contains exactly the experiment as
specified, and nothing else. Everything in `extras/` is additional work.

```
velpic/
  main/                 <- the experiment as asked for: orig | velpic | resid
    velpic_main.py
    RESULTS.md          <- read this one
    outputs/{mnist,cifar10,imagenet}/
  extras/
    controls/           rand, classmean, resid_orth, robust reference models
    memorization/       the paper's own regime: our own models trained on
                        scrambled labels
    adversarial/        PGD attacks on each head
  lib/
    velpic_core.py      shared machinery, imported by everything above
```

## The three steps

1. Build the VeLPIC class vectors from the penultimate layer
2. Replace the pre-softmax weights with them
3. Take the gradient of the prediction w.r.t. the input, and draw it

## Heads

| Head | What it is | Where |
| --- | --- | --- |
| `orig` | the trained weights, unchanged | main |
| `velpic` | the VeLPIC vectors, `W_velpic` | main |
| `resid` | the vector subtraction, `W_orig - W_velpic` | main |
| `resid_orth` | scale-free residual, `w_j - (w_j . v_j) v_j` | extras/controls |
| `rand` | random unit vectors | extras/controls |
| `classmean` | normalised class means | extras/controls |

## What is reused, what is new

The VeLPIC maths is the authors' own code, called unmodified from
`../Dynamics_during_training_DNN`:
`classwise_data`, `Added_data_layer`, `PCA_new_layer`, `avg_neg`.

Gradients and plotting come from `../visualize_prediction_gradients.py`, also
unmodified.

The only genuinely new code is a forward hook to read the penultimate layer,
`layer.weight.copy_(W)` to swap the head, the PGD loop, and the corrupted-label
training loop. Neither the authors' repo nor the pre-existing gradient script
was edited.

## Running

```
cd main
python -u velpic_main.py --dataset mnist
python -u velpic_main.py --dataset cifar10
python -u velpic_main.py --dataset imagenet
```

Use `python -u`, or output stays buffered and long runs look stalled.

ImageNet here means **Imagenette**, a 10-class subset of real ImageNet at
224x224. Full 1000-class ImageNet needs 150 GB of training data. Download it
once with:

```
curl -L -o ../../data/imagenette2-320.tgz \
  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf ../../data/imagenette2-320.tgz -C ../../data/
```

## Full write-up

`../VELPIC_REPORT.md` has the complete background, method, every experiment and
all observations. `main/RESULTS.md` is the short version covering only the
three heads above.
