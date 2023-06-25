# RegNeRF

<h4>Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs</h4>

```{button-link} https://github.com/google-research/google-research/tree/master/regnerf
:color: primary
:outline:
Code
```

```{button-link} https://arxiv.org/abs/2112.00724.pdf
:color: primary
:outline:
Paper
```

## Running the Model

```bash
ns-train regnerf
```

## Method

RegNeRF regularizes training to learn sparse (few input views) scenes better.
Three additional methods are added to the training process.

### Depth Smoothness Loss

One observation in the RegNeRF paper is that problems in sparse scenes usually
arise from messy geometry. While ground truth input views are learned well, the
underlying geometry is not "3D consistent", leading to messy reconstructions of
other views.

To address this, RegNeRF penalizes geometry that is not smooth, assuming that
"piece-wise smooth" structures are more common.

To create a loss function, these steps are followed:

- A number of "unobserved viewpoints" are randomly sampled from all possible
  views before training starts.
- At each training step:
  - A batch of unobserved viewpoints are selected.
  - The NeRF renders a patch (8x8 grid of pixels) from this viewpoint.
  - The loss is how "unsmooth" the depth map of the patch is, computed with the
    MSE of neighboring pixels.

### Color Likelihood

This aims to improve RGB color accuracy from unobserved viewpoints with a
measure of how likely a set of colors is to appear. Using the same patches as
above (Depth Smoothness Loss), the RGB image rendered from those viewpoints are
compared to "diverse natural images", incentivizing higher probability patches
(through the loss function).

A separate RealNVP model is trained on the "diverse natural images", learning
the likelihood of any 8x8 patch of pixels appearing. Renders from the patch
viewpoints are passed through, and a negative log likelihood loss is applied.

In both the official and this (CognitiveStudio) implementation, Color Likelihood
loss is omitted.

### Sample Space Annealing

Similar to depth smoothness, this aims to create a "3D consistent"
reconstruction.

To prevent divergence at the beginning of training, the near and far planes of
each ray are brought close together, and gradually expanded.
