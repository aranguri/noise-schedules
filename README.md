# Code for Optimizing Noise Schedules of Generative Models in High Dimensionss
This is the official repo for the experiments from the paper [Optimizing Noise Schedules of Generative Models in High Dimensionss](https://arxiv.org/abs/2501.00988) by S. Aranguri, G. Biroli, M. Mezard, E. Vanden-Eijnden (under review).

## Generating samples from a Gaussian mixture and Curie-Weiss distributions using exact velocity field
In the `exact_models` folder, we implemented a flow-based model to generate samples from a Gaussian mixture and the Curie-Weiss distributions using both the Variance Preserving (VP) and Variance Exploding (VE) schedules. Since this data distributions are simple, the velocity field can be obtained exactly. This is used to numerically verify and illustrate the claims in the paper.

## Comparing VP and VE schedules using CelebA dataset
In the `celeba_task` folder, we provide the code to generate samples using the VP and VE SDEs from Song et al 2020 [2] pre-trained on the CelebA-HQ dataset, with different number of discretization steps (under `celeba_task/run_vp.py` and `celeba_task/run_ve.py`). We then run a discriminator to measure the quality of the high- and low-level features on the generated images  (under `celeba_task/run_high.py` and `celeba_task/run_low.py`.) Finally, the code at `celeba_task/present_high.ipynb` and `celeba_task/present_low.ipynb` is used to make the following plots 

<p align="center">
  <img src="celeba_task/high.png" alt="Error" width="45%">
  <img src="celeba_task/low.png" alt="Active" width="45%">
</p>

We then see that the VE schedule outperforms the VP in the high-level aspects while this is reversed (when using a small number of steps) for the quality of the details of the generated image.

## a
In the scForked and modified Song et al implementation of score-based diffusion models to show that the variance exploding (VE) schedule outperforms the variance preserving (VP) one in the quality of the details of the generated image while this is reversed for the high-level aspects. This required modifying the noise schedules and implementing a discriminator to test the quality of the generated images.

## References
[1]
[2]
