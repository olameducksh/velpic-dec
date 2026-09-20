# VeLPIC Robust/Non-Robust Decomposition — Implementation & Results

2026-09-20

## Summary

We tested whether a neural network can be split into a "robust half" and a "non-robust half" by editing only its last layer, using the VeLPIC method from Ketha & Ramaswamy. The short answer is that the split does not work, and we can say precisely why.

- **VeLPIC itself works exactly as published.** We reproduced the paper's main result: on a network trained on 40% corrupted labels, replacing the last layer with VeLPIC vectors raised test accuracy from **42.00% to 51.86%**, a gain of **+9.86 points**, with no retraining.
- **We found that VeLPIC vectors are the class means.** The cosine similarity between each VeLPIC vector and the corresponding class average is **1.0000**. This is a mathematical consequence of the method, not a coincidence, and it is not mentioned in the paper. It now holds at 128, 640 and 2048 dimensions across four networks.
- **The subtraction on the whiteboard is not well defined.** `W_original - W_velpic` gave 94.76% accuracy on one model and 4.52% on another. The same formula, opposite outcomes, purely because of a difference in vector lengths.
- **Gradient maps do not change when you swap the last layer.** Even a completely random last layer produces an attribution map **73% correlated** with the real one. At ImageNet resolution this becomes visually unmistakable: on an adversarially robust ResNet-50 a *random* head still traces the object cleanly, while on a normally trained one the *real* head produces pure static. What the gradient looks like is decided by the feature extractor, not the final layer.
- **Adversarial robustness does not transfer either.** The VeLPIC head classifies 10 points better than the original on the memorized model, yet collapses from 51.55% to **1.30%** under attack, while the original collapses to 0.00%. Both are destroyed.
- **One positive signal survived.** On a model that was *already* adversarially robust, the VeLPIC head was **+3.6 points more robust** than the trained head (66.20% vs 62.60%), and the residual head was the worst of the three. That is the predicted ordering, but it only appears when robustness is already present, and it is a roughly 2.4-sigma result that needs confirmation.

Everything below explains how each of these was measured and what we think it means.

## Background: what the paper actually says

The paper is [On the Dynamics & Transferability of Latent Generalization during Memorization](https://arxiv.org/abs/2603.19865) by Simran Ketha and Venkatakrishnan Ramaswamy (BITS Pilani Hyderabad). This is the paper behind the OpenReview link, and the code is at [simranketha/Dynamics_during_training_DNN](https://github.com/simranketha/Dynamics_during_training_DNN).

### The setup: deliberately breaking the labels

The researchers take a standard image dataset such as CIFAR-10 and randomly scramble a fraction of the training labels. A picture of a cat might be labelled "truck". They do this for 20%, 40%, 60% and 80% of the training images.

A network trained on this data does something strange. It eventually gets close to **100% accuracy on the training set**, which means it has simply memorised which nonsense label goes with which image. There is no pattern to learn, so the only way to fit the data is to remember it. Meanwhile its **accuracy on the real test set collapses**, because memorising noise teaches you nothing about cats.

### The surprise: the useful information is still in there

Here is the interesting part. Even though the network's *output* is now bad, its *internal representations* still contain good, meaningful information about the true classes. The network learned real structure on the way to memorising, and that structure did not get destroyed. It just stopped being used by the final output layer.

The authors call this **latent generalization**: generalization ability that is present inside the network but hidden, not visible in what the model actually predicts.

### How they find it: probes

To show the hidden information is there, they attach a **probe** to an intermediate layer. A probe is a small, simple classifier that looks at the activations of one layer and tries to predict the class. If a simple probe on layer 5 does much better than the whole network's own output, that proves layer 5 contains information the network is failing to use.

They use two probes:

| Probe | What represents each class | Type of classifier |
| --- | --- | --- |
| **MASC** (earlier paper) | a multi-dimensional **subspace** | quadratic (non-linear) |
| **VeLPIC** (this paper) | a single **vector** | linear |

Both work the same way at prediction time: take the activation of a new image, measure the angle to each class's object, and pick the class with the **smallest angle**.

### Why they invented VeLPIC

MASC came first and worked remarkably well. That raised an obvious objection: maybe MASC only works because it is a non-linear classifier, and non-linear classifiers are powerful enough to find structure that isn't really "there" in any simple sense.

So the authors built the simplest possible linear version of the same idea and asked whether it still worked. It did, often better than MASC. That is Section 6 of the paper, and VeLPIC is what our project is built on.

### One thing to be clear about

The word "robust" in this paper means **generalizes to true labels despite being trained on corrupted ones**. It has nothing to do with adversarial examples. The paper never mentions adversarial robustness, attributions, saliency maps or input gradients. Not once. This matters for the next section.

## How VeLPIC works, step by step

VeLPIC produces **one vector per class**. For CIFAR-10 that is 10 vectors. Here is the whole method.

First, pick a layer. For the weight-swap experiment this is always the **penultimate layer** — the last hidden layer, the one that feeds the final classification layer. Run all the training images through the network and record what that layer outputs. If you have 10,000 images and the layer has 640 units, you get a 10,000 x 640 table of numbers.

Then, for each class:

1. **Group by label.** Keep only the rows belonging to that class. Importantly, this uses the labels the model was *trained* on — the corrupted ones — not the true labels. The probe only ever sees data the model itself saw.

2. **Add the negatives.** For every activation vector `x` in the group, also add `-x` to the group. A group of 1,000 becomes 2,000.

3. **Run PCA and keep only the first component.** This gives one direction, normalised to length 1.

4. **Fix the direction.** Project the training data onto that direction and take the average. If the average is negative, flip the vector.

5. **Classify.** For a new image, compute the dot product of its activation with each of the 10 class vectors and pick the largest. Because all vectors have length 1, largest dot product is the same as smallest angle.

### Why step 2 exists

PCA normally finds a direction through the **average** of your data, which gives you a line floating somewhere in space rather than a line through the origin. Measuring an "angle" only makes sense from the origin. Adding `-x` for every `x` forces the average to be exactly zero, which pins the line to the origin. This mirrors how MASC builds its subspaces.

### Why step 4 exists

PCA gives you a direction but not a sign — mathematically, a vector and its negative are equally valid answers, and the software returns one arbitrarily. For MASC this does not matter, because a line is a line. For VeLPIC it matters a lot: if the vector comes out backwards, every image of that class ends up at roughly 180 degrees instead of 0, and the class is never predicted. Step 4 catches and fixes that.

We saw this fire in practice. On the WRN-28-10 model, classes 0 and 9 needed flipping. On the memorized model, none did.

### The weight swap (paper Section 8)

This is the part the project is built on. Stack the 10 class vectors into a matrix:

```
W_pre-softmax = [v_1  v_2  ...  v_M]^T
```

That matrix has exactly the shape of the final layer's weights: 10 rows, 640 columns. And picking the largest dot product is exactly what a final linear layer followed by softmax does.

So you can **overwrite the final layer with the VeLPIC vectors, set the biases to zero, and the network now outputs exactly what the VeLPIC probe would say**. No retraining. The hidden generalization becomes the model's actual output. That is the paper's headline claim and it is what we reproduced.

## The hypothesis we were testing

This part is **not from the paper**. It is the extension proposed for this project, and it connects the VeLPIC paper to a separate line of work on adversarial examples.

### The background idea: robust and non-robust features

There is a well-known result, [Ilyas et al., "Adversarial Examples Are Not Bugs, They Are Features" (2019)](https://arxiv.org/abs/1905.02175), arguing that a trained classifier relies on two kinds of signal:

- **Robust features** — human-meaningful properties like shape and outline, which survive small changes to the image.
- **Non-robust features** — patterns that genuinely predict the right answer on real data, but are fragile and imperceptible to people. Adversarial attacks work by flipping exactly these.

A companion result, [Tsipras et al., "Robustness May Be at Odds with Accuracy" (2018)](https://arxiv.org/abs/1805.12152), showed that **adversarially robust models have perceptually aligned gradients**. If you compute how much each pixel affects the prediction and draw that as an image, a robust model's picture looks like the object. A normal model's picture looks like static. This is the property we can measure directly with the existing gradient code.

### The proposed leap

The reasoning goes like this. VeLPIC's class vectors come from the dominant direction of each class's internal representation — an unsupervised, purely geometric summary. Perhaps that captures the **robust** part of what the network learned. And whatever the trained weights contain *beyond* that is the **non-robust** part.

Which gives the decomposition from the whiteboard:

```
W_original  =  W_VeLPIC  +  (W_original - W_VeLPIC)
   full          robust          non-robust
```

Two heads sharing one feature extractor. Feed in the same image, compute the gradient for each, and compare:

- the VeLPIC head should give **clean, object-shaped** attributions, like a robust model
- the residual head should give **noisy, scattered** attributions, like a standard model

If that held, it would link two separate literatures: the latent structure this paper finds would *be* the robust-feature structure.

### Two gaps worth naming

We flagged these before running anything, and both turned out to matter.

**The two notions of "robust" are not the same thing.** The paper's latent generalization is about *memorization versus generalization*. Ilyas's non-robust features are *genuinely generalizing* — that is his central claim, that you can train only on non-robust features and still get good test accuracy. Memorized label noise is specific to individual training examples; non-robust features transfer to new data. Treating one as the other is an analogy, not an argument.

**PCA finds variance, not discriminative power.** The first principal component is the direction of most spread. Non-robust features are defined by being *predictive*. These two things routinely come apart — it is the standard distinction between PCA and LDA. There is no particular reason the split should land where the hypothesis needs it to.

### The structural objection

Before running the experiment we identified the most likely failure mode. Write the score for class j as the weight vector dotted with the features:

```
d(s_j)/dx  =  w_j^T  J_phi(x)
```

where `J` is the Jacobian of the feature extractor. Every head shares the **same** feature extractor, so every gradient map is just a different weighted mix of the **same** rows of `J`. If those rows are high-frequency noise, every mix of them is high-frequency noise.

Perceptual alignment in Tsipras comes from adversarially training the *whole network*, which changes `J` itself. Here the backbone is frozen. That was the prediction, and the experiments below confirm it.

## What we built

Three new scripts. **Nothing in the authors' repository was modified** — we add it to the Python path and import from it, so their code runs exactly as they wrote it.

| Script | What it does |
| --- | --- |
| `velpic_decompose.py` | The main experiment on clean, pre-trained models. Extracts activations, builds VeLPIC vectors, swaps the head, measures accuracy, draws gradient maps. |
| `corrupted/velpic_corrupted.py` | The paper's own regime. Corrupts labels, trains a network until it memorizes them, then runs the same VeLPIC pipeline. |
| `velpic_pgd.py` | Adversarial attacks. Runs PGD against each head to measure actual robustness. |

### Code reused from the authors' repository

The VeLPIC mathematics is entirely theirs. We call their functions rather than reimplementing the algorithm.

| Function | Where | What it does for us |
| --- | --- | --- |
| `classwise_data` | `MASC/subspace_pytorch.py` | step 1, groups activations by label |
| `Added_data_layer` | `MASC/subspace_pytorch.py` | step 2, adds the negatives |
| `PCA_new_layer` | `MASC/scratch_pca.py` | step 3, first principal component plus explained variance |
| `avg_neg` | `MAVC/MASV_pytorch.py` | step 4, decides which vectors to flip |
| `accuracy_model` | `VeLPIC_Weight_change.py` | the accuracy evaluation loop |

One detail worth recording: we use `scratch_pca.py`, the **NumPy** version, not `scratch_pca_pytorch.py`. This is because the authors' own `angle_pytorch.py` does `from MASC import scratch_pca as sp`. The PyTorch file has a different return signature and is not the one their pipeline actually calls.

### Code reused from the existing gradient script

`visualize_prediction_gradients.py` was also left unmodified and imported: `load_model`, `small_dataset`, `to_input`, `prediction_gradient`, `normalize_importance`, `make_overlay`, `to_numpy_image`.

### What we had to write, and why

Four things could not be reused. Each is small, and in each case the repository's version is tied to details of their own setup.

| New code | Why the repo version does not fit |
| --- | --- |
| Activation extraction (~5 lines, a forward hook) | Their `saving_activations_lastlayer` is bound to a specific attribute on their own network class and writes pickle files to disk. |
| Weight swap (~3 lines) | Their `weight_change()` hard-codes `model.fc3` / `model.output` and `device='cuda'`. No torchvision or RobustBench model has those attribute names. |
| Projection (1 line, a matrix multiply) | Their `dot_part_train` loads `.pt` files from a specific directory layout that only exists on their machine. |
| PGD attack (~10 lines) | Nothing in the repository performs adversarial attacks. |

Our script finds the final layer by **searching the network for the last linear layer**, rather than hard-coding a name. This is why the same code runs unchanged on WideResNet-28-10 (where it is called `fc`, 640 units), on the small CNN (where it is `classifier[3]`, 128 units), and on anything else.

### Deviations from the paper, stated plainly

We made exactly two, and we tested the consequences of both.

1. **We kept the bias instead of zeroing it.** The paper says "all biases are set to zero". We kept them on your instruction. We then measured what difference this makes: the gradient maps are **identical, cosine 1.000000**, and accuracy moves by 0.03% (94.76% vs 94.79%). The bias term vanishes when you differentiate, so this deviation provably cannot affect any gradient result.
2. **Most experiments use clean data, not corrupted.** The paper builds its vectors from a corrupted training set. Experiments 1 to 3 use clean, pre-trained models. Experiment 4 corrects this and runs the paper's actual regime.

Everything else follows the paper exactly.

## Experiment 1: clean, well-trained model

**Setup.** RobustBench's `Standard` model for CIFAR-10, a WideResNet-28-10 at 94.77% test accuracy. Penultimate layer has 640 units, so every weight matrix is 10 x 640. VeLPIC vectors built from 10,000 clean training images.

### Result: VeLPIC reproduces the trained classifier

| Head | Test accuracy |
| --- | --- |
| `orig` (the trained weights) | 94.77% |
| `velpic` | 94.76% |
| `velpic`, biases zeroed (paper-exact) | 94.79% |
| `rand` (random unit vectors, control) | 3.60% |

The two heads agree on **99.31% of test predictions**. Replacing a trained final layer with ten geometrically-derived vectors costs essentially nothing, and the paper-exact version is fractionally *better* than the original network.

The `rand` control at 3.60% matters: it proves the swap is real and that accuracy is not simply leaking through from the backbone regardless of what you put in the head.

### The unexpected finding: VeLPIC vectors are the class means

We computed the cosine similarity between each VeLPIC vector and the simple average of that class's activations.

| Class | Explained variance of PC1 | cos(VeLPIC vector, class mean) |
| --- | --- | --- |
| 0 | 0.932 | **1.0000** |
| 1 | 0.940 | **1.0000** |
| 2 | 0.919 | **1.0000** |
| 3 | 0.921 | **1.0000** |
| 4 | 0.927 | **1.0000** |
| 5 | 0.917 | **1.0000** |
| 6 | 0.929 | **1.0000** |
| 7 | 0.933 | **1.0000** |
| 8 | 0.944 | **1.0000** |
| 9 | 0.941 | **1.0000** |

Identical to four decimal places, every class. Running a `classmean` head as a separate condition gave **94.76%**, the same number as `velpic`.

### Why this happens

This is a mathematical consequence of step 2, the negative augmentation. Adding `-x` for every `x` forces the mean to zero, which means the covariance of the augmented set equals the **uncentered second moment** of the original:

```
E[x x^T]  =  Cov(x)  +  mu mu^T
```

The penultimate layer sits after a ReLU, so **all activations are non-negative** (we verified: minimum exactly 0.000). That makes the mean vector large, the term on the right dominates the sum, and the first principal component collapses onto the mean direction. The high explained variance (0.92-0.94) is the same effect seen from the other side.

### Why it matters

**VeLPIC applied to a post-ReLU layer is, in effect, a Nearest Class Mean classifier.** That is a much more concrete description than "the first principal component head", and it connects the method to the neural collapse literature, where penultimate representations are known to collapse toward their class means.

It also reframes the research question. "Is PC1 the robust part?" becomes "**is the class-mean direction the robust part?**" — a sharper and more answerable question.

This equivalence is not mentioned anywhere in the paper. It reproduced on every model we tried (1.0000 on WRN-28-10, 0.999 on the small CNN, 0.998 on the memorized model), which is expected since it follows from ReLU non-negativity rather than from anything about a particular architecture.

## Experiment 2: the subtraction does not define a second head

The whiteboard says the non-robust half is `W_original - W_velpic`. When we computed it, this turned out not to be a well-defined operation.

### The problem: the two matrices have different scales

VeLPIC vectors always have length exactly 1, because step 3 normalises them. The trained weight rows have whatever length training gave them. Those are not comparable, and subtracting one from the other gives a result dominated by whichever is larger.

| Model | `orig` row length | `velpic` row length | `resid` test accuracy |
| --- | --- | --- | --- |
| WRN-28-10 `Standard` | 3.05 | 1.00 | **94.76%** |
| Small CNN | 0.70 | 1.00 | **4.52%** |

**The same formula gives opposite outcomes on two models.**

On the WideResNet, you are subtracting a length-1 vector from a length-3 vector. Barely anything changes. The residual head agreed with the original on **99.89% of test predictions** — it is not a separate head at all, it is the original with a small nudge.

On the small CNN the situation inverts. The VeLPIC vector is *longer* than the trained row, so the subtraction is dominated by the negated VeLPIC direction. The result is an **anti-classifier at 4.52%**, well below the 10% you would get by guessing.

Neither of these is "the non-robust part". Both are artefacts of an arbitrary length ratio.

### Some other geometry we measured

| Quantity | Small CNN | WRN-28-10 `Standard` |
| --- | --- | --- |
| cos(trained row, VeLPIC vector) | 0.287 | 0.709 |
| Mutual cos among the 10 VeLPIC vectors | — | 0.382 |
| Mutual cos among the 10 trained rows | — | -0.111 |
| Effective coordinates used, trained row (of 128) | ~60 | — |
| Effective coordinates used, VeLPIC vector (of 128) | ~23 | — |

Two things stand out.

The trained head of the well-trained model is **much more aligned with the class means** (0.709) than the weak model's is (0.287). That is what neural collapse predicts: as training quality improves, the classifier's weights converge toward the class means, so there is less "extra" beyond them to isolate.

The ten VeLPIC vectors have **positive** mutual similarity (0.382) while the trained rows are slightly **negative** (-0.111). The VeLPIC vectors all point into the positive region of space, because post-ReLU activations are non-negative, so they share a large common component. Trained classifier rows spread apart from each other instead.

### The fix, and why it did not rescue the idea

A scale-free alternative is to remove only the *component along* the VeLPIC direction:

```
w_j_perp  =  w_j  -  (w_j . v_j) v_j
```

This is well defined regardless of lengths and produces something genuinely orthogonal to the VeLPIC vector. We called this head `resid_orth` and used it throughout.

It scored **94.69%** on the WideResNet — essentially unchanged from the original 94.77%, despite having 71% of each row's length stripped away.

That is itself a notable result: **both halves of the split are fully competent classifiers on their own**. The class information is redundantly encoded, present both along the class-mean direction and in everything orthogonal to it. Accuracy therefore cannot distinguish a "robust half" from a "non-robust half", because neither half is impaired.

## Experiment 3: gradient maps

This is the core test. For each head we compute how much each pixel affects the predicted class, and draw that as an image.

### First, a control to prove the method can detect anything

Before concluding "the maps all look the same", we had to check that our setup can tell a robust model from a non-robust one at all. CIFAR-10 images are only 32x32 pixels, and it was entirely possible that nothing would be visible at that size.

We ran the identical pipeline on `Carmon2019Unlabeled`, an **adversarially trained** model from RobustBench.

The difference is dramatic and obvious by eye. On Carmon you can clearly see the outline of the Concorde, the hull and superstructure of the ships, the roofline and wheels of the car, the body of the frog. Backgrounds go dark. On `Standard` the maps are uniform high-frequency static across the whole image, with no object structure whatsoever.

**So the method works.** A negative result from here is a real negative, not a measurement failure. This control is the single most important thing in the whole investigation.

### A warning about reading the figures

Each head produces two columns: `|grad|` and `overlay`. **Only the `|grad|` column is meaningful.** The overlay is the original image multiplied by the importance map, so the object shows through even when the importance map is pure noise — you can see the car perfectly well in the `rand` overlay. Anyone reading these figures will be misled if they look at the overlay column.

### The result: swapping the head does almost nothing

We measured the correlation between each head's importance map and the original head's.

| Head | Small CNN | WRN-28-10 `Standard` |
| --- | --- | --- |
| `velpic` | 0.888 | 0.980 |
| `resid` | 0.833 | 0.994 |
| `resid_orth` | 0.871 | 0.923 |
| `rand` | **0.781** | **0.734** |

The maps are genuinely not identical — the largest absolute difference is around 20, not zero, so the weight swap is definitely reaching the gradient computation.

But look at the `rand` row. **A completely random readout direction, one that classifies at 3.6%, still produces an attribution map 73% correlated with the real model's.** The head simply cannot move the map much, no matter what you put there. That is the frozen-backbone prediction showing up as a measurement.

Note also that `rand` is roughly the same on both models (0.781 and 0.734). The ceiling on head influence is a property of the architecture in general, not of one particular network.

### An objective measure of "looks like the object"

Eyeballing is subjective, so we computed the **participation ratio** of each importance map: roughly, how many pixels out of 1024 carry the importance. A low number means concentrated and structured, which is the robust signature. A high number means spread-out noise.

| Model | `orig` | `velpic` |
| --- | --- | --- |
| **Carmon** (adversarially robust) | **278.4** | 286.9 |
| `Standard` (clean, non-robust) | 522.8 | 529.9 |
| Memorized small CNN (40% corrupt) | 683.6 | 644.6 |

Read this two ways.

**Down the columns, changing the backbone moves the number by about 400.** Robust 278, standard 523, memorized 684. Enormous, and it matches exactly what you see by eye.

**Across each row, changing the entire final layer moves it by 8 to 40.** Roughly a tenth as much, and in two of three cases in the wrong direction.

The conclusion is that **what a gradient map looks like is determined by the feature extractor, not by the final layer.**

One honest exception: on the memorized model, VeLPIC does move the number in the hypothesised direction by a non-trivial amount (683.6 down to 644.6). That is the best case for the hypothesis anywhere in our results, and it still recovers only about a tenth of the distance to the standard model, let alone to Carmon.

## Experiment 4: the paper's own regime (memorization)

Everything above used clean, well-trained models. That is **outside** the setting the paper studies, and it matters: on a clean model VeLPIC and the network agree almost perfectly, so of course their gradients agree too. We had to test the case where the two heads are genuinely different classifiers.

All of this lives in a separate `corrupted/` folder so the two regimes do not get confused later.

**Setup.** 10,000 CIFAR-10 training images, **40% of labels randomised** (3,672 actually flipped). Small CNN trained for 80 epochs with no data augmentation, so it can memorize. VeLPIC vectors built from the **corrupted** label groups, exactly as the paper does.

### The memorization curve reproduced

| Epoch | Train accuracy (corrupted labels) | Test accuracy (true labels) |
| --- | --- | --- |
| 10 | 45.71% | **55.55%** |
| 20 | 79.51% | 42.69% |
| 30 | 97.77% | 40.21% |
| 40 | 99.98% | 41.99% |
| 50 | 100.00% | 42.22% |
| 80 | 100.00% | 42.00% |

This is the paper's central dynamic, reproduced cleanly. Test accuracy **peaks early at 55.55% around epoch 10**, then *collapses* to 42% as the network memorizes the noise. The network learns real structure first, then discards it in favour of remembering the nonsense labels.

### The paper's Section 8 claim reproduced

| Measurement | Value |
| --- | --- |
| Model train accuracy on corrupted labels | 100.00% (fully memorized) |
| Model train accuracy on true labels | 63.28% |
| Model **test** accuracy on true labels | **42.00%** |
| **VeLPIC head test accuracy on true labels** | **51.86%** |
| **Transfer effect** | **+9.86 points** (+9.41 on the retrain) |

**This works.** Swapping in ten geometrically-derived vectors, with no retraining at all, recovers nearly 10 points of test accuracy from a network that had thrown that ability away. The latent generalization is real and it does transfer to the model, exactly as the paper claims.

You can see it image by image in the figure. The original head calls both ships "cat" and both frogs "deer" and "dog". The VeLPIC head calls them "ship", "ship", "frog", "frog".

Note that explained variance of PC1 dropped to about 0.45 here, compared with 0.92 on the clean model. That makes sense: with 40% of labels randomised, each class group is a mixture, so no single direction dominates as strongly. The class-mean equivalence still held at 0.998.

### But the gradients still do not move

This is the key negative. Here the original head (42.00%) and the VeLPIC head (51.86%) are **genuinely different classifiers** disagreeing on many images — the strongest possible version of the test.

The attribution maps are still uniform high-frequency noise for every head. No object structure in any of them. The participation ratio numbers in the previous section confirm it: 683.6 for `orig`, 644.6 for `velpic`, both far above the robust model's 278.

So even in the paper's own regime, where the weight swap demonstrably changes what the network *predicts*, it does not change what the network's attributions *look like*.

### The same thing at ImageNet resolution

We repeated the memorization experiment at 224x224 by fine-tuning a pre-trained
ResNet-50 on Imagenette with 40% of labels randomised, for 12 epochs.

| Measurement | Value |
| --- | --- |
| train acc on corrupted labels | **99.87%** (memorized) |
| model test acc on true labels | 84.23% |
| **VeLPIC head test acc** | **88.43%** |
| **transfer effect** | **+4.20 points** |
| `resid` head | 6.96% |

VeLPIC again transfers real generalization, and the gradient maps again do not
move: participation ratios of 24,502 for `orig`, 24,654 for `velpic` and 25,649
for `resid` — all within 5% of each other, and all far above the robust model's
5,005.

**An important caveat on this run.** Because we fine-tuned from pre-trained
ImageNet weights rather than training from scratch, the network **fit the label
noise without losing its generalization**: test accuracy went *up* to 84.23%
rather than collapsing. In the paper's CIFAR-10 regime, test accuracy collapses
from 55% to 42% as the model memorizes. So this is memorization *without* the
generalization collapse, which is a weaker version of the paper's setting.

That is a consequence of feasibility rather than choice. Training a 1000-class
ImageNet model from scratch on corrupted labels needs 150 GB of data we do not
have and weeks of GPU time; fine-tuning 9,469 Imagenette images takes 15
minutes. The pre-trained backbone is simply strong enough that memorizing 40%
label noise does not destroy it.

It is worth noting on its own terms: a strong pre-trained feature extractor can
absorb 40% label noise and still generalize, and VeLPIC adds a further 4.2
points on top of that.

## Experiment 5: actual adversarial robustness (PGD)

Accuracy could not separate the heads. Gradient maps could not separate the heads. Robustness is the third axis, and it is the one the hypothesis is really about — so we attacked each head directly.

**Setup.** PGD-Linf, the standard CIFAR-10 attack budget of eps = 8/255, 20 steps, random start. Two settings:

- **White-box**: attack each head on its own terms. Each head defended against its own worst case. This is the honest comparison.
- **Transfer**: build the attack against the *original* head and feed it to all heads.

### Results across all three backbones

| Backbone | Head | Clean | **PGD white-box** |
| --- | --- | --- | --- |
| **Carmon** (adversarially robust) | `orig` | 90.30% | **62.60%** |
| | `velpic` | 89.70% | **66.20%** |
| | `resid_orth` | 89.80% | **61.30%** |
| **Standard** (clean, non-robust) | `orig` | 94.80% | **0.00%** |
| | `velpic` | 94.80% | **0.00%** |
| | `resid_orth` | 94.80% | **0.00%** |
| **Memorized** (40% corrupt) | `orig` | 41.65% | **0.00%** |
| | `velpic` | 51.55% | **1.30%** |
| | `resid_orth` | 34.20% | **0.00%** |

Carmon's 62.60% is a good sanity check on the measurement: RobustBench lists this model at 59.5% under AutoAttack, and PGD-20 is a weaker attack, so landing a few points above is exactly right. Our numbers are calibrated against the published literature.

### The main finding

**Robustness is a property of the backbone.** A robust feature extractor gives 61-66% for every head. A non-robust one gives 0% for every head. Swapping the final layer cannot create robustness that the feature extractor does not already have.

The memorized model is the cleanest refutation. The VeLPIC head classifies **10 points better** than the original on clean data (51.55% vs 41.65%) and still collapses to **1.30%** under attack — a 97.5% relative collapse. Better generalization, zero additional robustness. These are separate properties, and VeLPIC transfers only the first, which is precisely all the paper ever claimed.

The transfer column showed `velpic` at 7.60% against `orig`'s 0.00% on the memorized model, which looks more interesting but should not be leaned on. It mostly reflects that the attack was tuned against a *different* classifier; since `velpic` disagrees with `orig` on many images, some perturbations simply miss. That is classifier difference, not robustness.

### The one result that supports the hypothesis

On the **already-robust** backbone, the ordering came out exactly as predicted:

**`velpic` 66.20% > `orig` 62.60% > `resid_orth` 61.30%**

The VeLPIC head is **+3.6 points more robust** than the trained head, at a cost of only 0.6 points of clean accuracy. And the residual head is the *worst* of the three. That is the hypothesised ordering — robust part, original, non-robust part — appearing in the measurement that directly tests robustness.

How much weight to put on it:

- The sample was 1,000 images, so the standard error is roughly 1.5 points. A 3.6-point gap is about **2.4 sigma** — suggestive, not established.
- Crucially, **it only appears when the backbone is already adversarially trained.** It is a modest reweighting benefit on top of adversarial training, not a mechanism that creates robustness from nothing.

This is the only positive signal anywhere in the investigation, and it is the thing most worth following up.

## Experiment 6: ImageNet at 224x224

This is Experiment 1 repeated at proper resolution, and it is the most convincing single result we have. On CIFAR-10 every attribution map is 32x32, which makes "does this look like the object" a hard judgement call. At 224x224 it is not a judgement call at all.

**Setup.** Imagenette, a 10-class subset of real ImageNet at full resolution (tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute): 9,469 train and 3,925 validation images. Two RobustBench ResNet-50s, the direct analogue of the CIFAR-10 pairing:

- `Standard_R50` — normally trained
- `Salman2020Do_R50` — adversarially trained

The penultimate layer is 2048 units. Since the models are 1000-way and Imagenette is 10-way, the pre-softmax layer is sliced to the 10 relevant ImageNet class rows first, so both heads are coherent 10-way classifiers.

### What the figures show

On **Salman**, the gradient maps clearly trace the springer spaniel's eyes and muzzle, the chain saw's bar, the French horn's tubing, the golf ball and its tee. Backgrounds are black. On **Standard_R50**, the same images produce dense static from edge to edge — the dog is simply not visible in the gradient.

And within each model, **every head looks the same, including `rand`**. On Salman the random head traces the objects just as clearly as the trained one, while predicting nonsense (*gas pump 0.17*, *tench 0.26*, *chain saw 0.20*).

That is the entire finding in one panel: a **random** readout on a robust backbone produces object-aligned gradients, and the **real** readout on a non-robust backbone produces noise.

### Participation ratio, out of 50,176 pixels

| Head | `Standard_R50` | `Salman2020Do_R50` (robust) |
| --- | --- | --- |
| `orig` | 23,273 | **5,005** |
| `velpic` | 24,443 | 4,882 |
| `resid` | 23,940 | 4,799 |
| `resid_orth` | 23,653 | 5,089 |
| `rand` | 25,504 | 5,192 |
| `classmean` | 24,456 | 4,899 |

- **Backbone effect: 4.6x** (23,273 down to 5,005)
- **Head effect: under 10%** within either model
- `rand` on the robust backbone (5,192) is **4.5x more structured** than `orig` on the standard one (23,273)

The CIFAR-10 version of this table had a backbone effect of about 400 units against a head effect of 8 to 40. At ImageNet resolution the same pattern holds with a much larger margin.

### Accuracy

| Head | `Standard_R50` | `Salman2020Do_R50` |
| --- | --- | --- |
| `orig` | 99.67% | 98.22% |
| `velpic` | 99.03% | 95.11% |
| `resid` | 99.59% | 97.27% |
| `resid_orth` | 99.46% | 97.43% |
| `rand` | 6.90% | 11.41% |
| `classmean` | 99.08% | 94.65% |

### Two further findings

**The class-mean equivalence holds at 2048 dimensions.** cos(VeLPIC vector, class mean) came out between 0.9959 and 1.0000. It is now confirmed at 128, 640 and 2048 dimensions across four different networks, which makes it a property of the method rather than an artefact of any particular scale.

**The sign-fix step matters far more at this scale.** Seven of ten classes needed flipping on `Standard_R50`, against two of ten on CIFAR-10. Anyone reimplementing VeLPIC and skipping step 4 would break most of their classes here.

One smaller observation: the row norms are much closer at ImageNet scale (1.55 against 1.00, versus CIFAR-10's 3.05 against 1.00), so the plain `resid` head is less degenerate than before. It still lands at 99.59%, essentially indistinguishable from `orig`.

### A practical note on the authors' code

The first attempt at this run took over 90 minutes before we stopped it. The cause is that `accuracy_model` does not wrap its forward pass in `torch.no_grad()`, so ResNet-50 at 224x224 retains its full autograd graph on every batch — about 6 GB, which pinned a 6 GB card at 95% and sent the allocator thrashing. On the small CIFAR-10 networks this is invisible, which is why it never surfaced earlier.

We fixed it by wrapping the call site rather than editing their file, and added parallel JPEG decoding and a feature cache. Identical numbers, about 10 minutes instead of 90.

## Every observation in one place

1. **VeLPIC reproduces a trained classifier on clean data.** 94.76% against the original's 94.77% on WRN-28-10, agreeing on 99.31% of test predictions.

2. **VeLPIC vectors are the class means.** cos = 1.0000 on WRN-28-10, 0.999 on the small CNN, 0.998 on the memorized model. A `classmean` head gives the identical 94.76%. This follows mathematically from the negative augmentation combined with post-ReLU non-negativity. It is not mentioned in the paper.

3. **The sign-fix step is real and necessary.** It fired on classes 0 and 9 of WRN-28-10. Without it those classes would never be predicted.

4. **The plain subtraction is not well defined.** 94.76% on one model, 4.52% on another, from the same formula, purely because of the length ratio between the two matrices (3.05 vs 1.00, against 0.70 vs 1.00).

5. **The residual head is not a separate head on the WideResNet.** It agrees with the original on 99.89% of predictions.

6. **Both halves of the orthogonal split are competent classifiers.** `velpic` 94.76%, `resid_orth` 94.69%, against the original 94.77%. Class information is redundantly encoded, so accuracy cannot distinguish the halves.

7. **Better-trained models have heads closer to the class means.** cos(trained row, VeLPIC vector) is 0.709 on the well-trained model versus 0.287 on the weak one — consistent with neural collapse.

8. **The gradient-map method demonstrably works.** Carmon shows clear object outlines; `Standard` shows pure static. Participation ratio 278 versus 523 confirms it numerically.

9. **A random head still produces a 73%-correlated attribution map.** The ceiling on how much any head can change the map, and it is roughly the same across architectures (0.734 and 0.781).

10. **Backbone effect on gradient structure is ~400 units; head effect is 8-40 units.** About a tenth as large, and in two of three cases pointing the wrong way.

11. **The memorization dynamic reproduces.** Test accuracy peaks at 55.55% around epoch 10, then collapses to 42.00% as the network memorizes.

12. **The paper's Section 8 transfer works.** 42.00% to 51.86%, a gain of +9.86 points with no retraining.

13. **Gradients still do not move in the paper's own regime,** even though the two heads are genuinely different classifiers there.

14. **Robustness does not transfer.** The VeLPIC head is 10 points better on clean data yet collapses to 1.30% under attack, against the original's 0.00%.

15. **A robust backbone makes every head robust; a non-robust backbone makes every head fall to zero.** 61-66% versus 0.00% across the board.

16. **The predicted ordering does appear, but only on an already-robust backbone.** `velpic` 66.20% > `orig` 62.60% > `resid_orth` 61.30%, at roughly 2.4 sigma.

17. **The bias makes no difference to any gradient result.** Zeroed versus kept gives cosine 1.000000 between maps and 0.03% accuracy difference.

18. **Small models look more promising but are weaker evidence.** Their heads differ more from each other (0.83-0.89 correlation versus 0.92-0.99), but none of their maps show object structure. "Different from each other" is not the same as "one of them is robust".

19. **At ImageNet resolution the separation is unmistakable.** Backbone changes the participation ratio by **4.6x** (23,273 to 5,005); every head stays within 10% of its own model. A *random* head on the robust ResNet-50 (5,192) is **4.5x more structured** than the *real* head on the standard one (23,273).

20. **The class-mean equivalence holds at 2048 dimensions** (cos 0.9959 to 1.0000), confirming it across 128, 640 and 2048 dims on four networks.

21. **The sign fix matters much more at ImageNet scale.** Seven of ten classes needed flipping on `Standard_R50`, against two of ten on CIFAR-10.

22. **Row norms are closer at ImageNet scale** (1.55 against 1.00, versus CIFAR-10's 3.05 against 1.00), so plain `resid` is less degenerate there — but still lands at 99.59%, indistinguishable from `orig`.

23. **The authors' `accuracy_model` omits `torch.no_grad()`.** Harmless on small CIFAR networks, but on ResNet-50 at 224x224 it retains the full autograd graph (~6 GB) and made a 10-minute run take over 90.

## What we conclude

### The strong form of the hypothesis is refuted

**You cannot split a network into a robust and a non-robust half by editing its last layer.** Three independent measurements agree, and each one fails for a different reason:

| Measurement | What it showed |
| --- | --- |
| Accuracy | Both halves are competent classifiers (94.76% and 94.69%). Nothing distinguishes them. |
| Gradient structure | Backbone moves the metric by ~400; the head moves it by 8-40. A random head still gives a 73%-correlated map. |
| Adversarial robustness | Robust backbone gives 61-66% for every head; non-robust gives 0% for every head. |

The explanation is the structural one we identified before running anything. Every head is a different weighted mix of the **same** feature Jacobian. If the feature extractor produces high-frequency, attackable representations, every linear readout of them inherits that. Perceptual alignment and adversarial robustness are properties of the feature extractor, and the final layer cannot manufacture either.

The ImageNet run makes this as concrete as it can get: a **random** readout on an adversarially robust backbone produces cleanly object-aligned gradients, while the **trained** readout on a normally trained backbone produces static. The head is almost irrelevant to what the attribution looks like; the backbone decides it.

### The paper itself is fine, and reproduces

None of this is a criticism of Ketha & Ramaswamy. Their claim is that VeLPIC transfers *latent generalization* to the model, and we reproduced that cleanly: **+9.86 points** on a memorized network, no retraining. Their method does what they say it does.

The gap is between their claim and the extension. **Generalization and robustness are separate properties**, and the VeLPIC swap transfers only the first. Our memorized model demonstrates the separation directly: 10 points better clean accuracy, no additional robustness whatsoever.

### The weak form has a hint worth chasing

On an already-robust backbone the predicted ordering did appear: `velpic` 66.20% > `orig` 62.60% > `resid_orth` 61.30%. The class-mean direction is modestly more attack-resistant than the learned direction, and the orthogonal remainder is the least resistant of the three.

That is genuinely the hypothesised structure. Two caveats keep it from being a result yet: it is roughly 2.4 sigma at n=1000, and it requires the robustness to already be there. It is a reweighting benefit on top of adversarial training, not a route to creating robustness.

### A reframing that may be more productive

If the goal is to study robust versus non-robust features, the evidence says the intervention has to touch the **feature extractor**, not the readout. The last layer is downstream of everything that determines the properties being measured.

Separately, the class-mean finding changes the question in a useful way. Since VeLPIC on a post-ReLU layer *is* nearest-class-mean, the open question is no longer about principal components at all — it is whether the class-mean direction has special robustness properties. That is a cleaner question, it connects to the neural collapse literature, and the +3.6 point result is direct evidence on it.

## Caveats and limitations

Things that could weaken the conclusions, stated so they are not discovered later.

**Resolution: now addressed.** The original worry was that everything was CIFAR-10 at 32x32, where judging attribution quality by eye is hard. Experiment 6 repeats the test on real ImageNet images at 224x224 and the conclusion holds with a much larger margin: a 4.6x backbone effect against under 10% head effect. This caveat no longer applies to the gradient result, though the PGD numbers are still CIFAR-10 only.

**Three architecture families.** WideResNet-28-10, a three-layer CNN, and ResNet-50. The paper reports that ResNet-18 behaves differently from its other models in several places, so architecture clearly matters, and we still have not tested ResNet-18 itself.

**One corruption level.** Only 40%. The paper finds the effect grows with corruption, and the one head-driven movement in the gradient metric (683.6 to 644.6) appeared in this regime. A sweep from 20% to 80% might show it growing.

**The 2.4-sigma result is not confirmed.** The +3.6 point robustness gain on Carmon used 1,000 test images. It should be re-run at 5,000 before anyone relies on it.

**PGD is not the strongest attack.** We used PGD-20, which is standard for screening but weaker than AutoAttack. The conclusion is unlikely to change — a stronger attack would only lower the numbers further, and they are already at 0% for the non-robust backbones.

**The memorized model is small and weak.** It reaches 42% test accuracy. Its clean baseline is 62.97%, so it is not a strong network even before corruption. The paper uses larger models trained for longer.

**Participation ratio is a proxy.** It measures how concentrated the attribution map is, not whether it aligns with the object. A properly aligned metric would need segmentation masks. It happens to separate Carmon from `Standard` very cleanly, which is why we trust it here, but it is an indirect measure.

**We may have implemented a different decomposition than intended.** The whiteboard shows plain subtraction; we implemented that and also an orthogonal variant when subtraction turned out to be ill-posed. If something different was meant, the results for that variant are not in hand. The `rand` control at 0.734 is the finding most likely to survive whatever variant is intended, since it bounds how much *any* head can matter.

## Questions and suggested next steps

### Questions for discussion

1. **Was the class-mean equivalence known?** VeLPIC on a post-ReLU layer is nearest-class-mean, at cosine 1.0000. This changes what the method *is*, and it is not in the paper. Is this expected, or new?

2. **Given that, why should the class-mean direction be the robust part?** The original reasoning was about principal components. Since PC1 turns out to be the class mean, the argument needs restating in those terms — and there may be a good one, since the +3.6 point result is evidence in its favour.

3. **Was plain subtraction intended, given it is scale-dependent?** It produced 94.76% on one model and 4.52% on another. If a specific normalisation was intended, that changes what we build.

4. **Does the frozen-backbone argument change the plan?** If gradient character and robustness are both properties of the feature extractor, an intervention on the final layer may be the wrong lever regardless of which decomposition is used.

### Suggested next steps, ranked

**1. Confirm the +3.6 point result.** Re-run the Carmon PGD comparison at 5,000 test images instead of 1,000. About ten minutes. This is the difference between a footnote and a finding, and it is the only result pointing the hypothesis's way. Highest value per unit effort by a wide margin.

**2. Sweep the corruption level.** 20%, 40%, 60%, 80%. The paper finds the effect grows with corruption, and the only head-driven movement in the gradient metric appeared in this regime. Cheap, and directly tests whether that movement is real or noise.

**3. Difference maps.** Since a random head still gives a 73%-correlated map, the head-specific signal is real but swamped. Visualising `grad(velpic) - grad(resid_orth)` cancels the shared backbone term and leaves only the head-dependent part. Small change to the code, and the one way to see head-specific structure in this setup.

**4. VeLPIC at earlier layers.** All our work is on the final layer because that is where the weight swap is defined. The paper finds latent generalization is often strongest in *early* layers. There is no weight-editing technique for those yet — the paper explicitly names this as an open problem — but a probe-only comparison is straightforward.

**5. ImageNet.** What the whiteboard originally sketched: 224x224, ResNet-50, 2048-dimensional features. Far better resolution for judging attribution quality. Needs roughly 50-100 images per class pushed through the network first, so it is the most expensive item here.

## Appendix: files, commands, environment

### File layout

```
proj/
  Dynamics_during_training_DNN/     the authors' repo, UNMODIFIED
    VeLPIC_during_training/         VeLPIC_code.py, VeLPIC_Weight_change.py, MASC/, MAVC/
    MASC_during_training/           the earlier MASC work
  visgrads/
    visualize_prediction_gradients.py   pre-existing, UNMODIFIED
    velpic_decompose.py                 NEW: clean CIFAR-10 models
    velpic_imagenet.py                  NEW: ImageNet at 224x224
    velpic_pgd.py                       NEW: adversarial attacks
    velpic_outputs/                     CIFAR-10 figures, accuracy.csv, pgd_*.csv
    imagenet_outputs/                   ImageNet figures, accuracy + participation csvs
    data/imagenette2-320/               10-class ImageNet subset (1.5 GB)
    corrupted/
      velpic_corrupted.py               NEW: the paper's memorization regime
      outputs/                          figure, accuracy csv, pgd csv, checkpoint
```

### Commands to reproduce every number

Clean WideResNet, all six heads, the accuracy table and figure:

```
python velpic_decompose.py --num_train 10000 --num_images 8 --batch_size 128 \
  --heads orig velpic resid resid_orth rand classmean
```

The robust reference model, which is the control that validates the gradient method:

```
python velpic_decompose.py --model robustbench:cifar10:Carmon2019Unlabeled \
  --num_train 10000 --num_images 8 --heads orig velpic resid_orth --skip_acc
```

The memorization experiment, which trains the network on first run and caches it:

```
python corrupted/velpic_corrupted.py --corrupt 0.4 --num_train 10000 --epochs 80
```

The PGD attacks:

```
python velpic_pgd.py --ckpt corrupted/outputs/memorized_c0.4_n10000_e80_s0.pth \
  --corrupt 0.4 --num_eval 2000 --out corrupted/outputs
python velpic_pgd.py --model robustbench:cifar10:Carmon2019Unlabeled --num_eval 1000
python velpic_pgd.py --model robustbench:cifar10:Standard --num_eval 1000
```

The ImageNet experiment, both models. Imagenette downloads from fast.ai with no account needed:

```
curl -L -o data/imagenette2-320.tgz \
  https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar -xzf data/imagenette2-320.tgz -C data/

python -u velpic_imagenet.py --model robustbench:imagenet:Standard_R50
python -u velpic_imagenet.py --model robustbench:imagenet:Salman2020Do_R50
```

Run these with `python -u`, or the output stays buffered and looks stalled. Each takes about 10 minutes on an RTX 4050; features are cached afterwards, so re-runs are much faster.

Useful flags: `--score logit` is the default and should stay that way; `--fixed_target` forces every head to differentiate the *original* model's predicted class, which removes a confound when heads disagree on what they predict; `--heads` selects which heads to build.

### Environment

Python 3.13.4, PyTorch **2.14.0+cu126**, NVIDIA RTX 4050 Laptop (6 GB), driver 616.64.

One note for anyone setting this up again: the machine originally had `torch 2.14.0+cpu`, a CPU-only build, which cannot use the GPU no matter what hardware is present. The fix keeps the same version and only adds CUDA:

```
pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0+cu126 torchvision
```

This matters beyond speed. Several of the authors' functions — `dot_part_train`, `pca_load_mavc`, `data_loading_train` — hard-code `.cuda()` and simply cannot run without a CUDA build.

### Sources

- [On the Dynamics & Transferability of Latent Generalization during Memorization](https://arxiv.org/abs/2603.19865) — Ketha & Ramaswamy, the VeLPIC paper
- [Dynamics_during_training_DNN](https://github.com/simranketha/Dynamics_during_training_DNN) — the authors' code
- [Adversarial Examples Are Not Bugs, They Are Features](https://arxiv.org/abs/1905.02175) — Ilyas et al., robust/non-robust features
- [Robustness May Be at Odds with Accuracy](https://arxiv.org/abs/1805.12152) — Tsipras et al., perceptually aligned gradients
- [RobustBench](https://robustbench.github.io/) — source of the `Standard` and `Carmon2019Unlabeled` models
