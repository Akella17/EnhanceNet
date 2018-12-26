## EnhanceNet

This is an implementation of the paper ["EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis"](https://arxiv.org/abs/1612.07919) by *M. Sajjadi et al., (2017)*.

### Model Architecture

![](https://github.com/Akella17/EnhanceNet/raw/master/data/enhancenet_architecture.png | width = 100)

Fully convolutional network architecture for 4x super-resolution which only learns the residual between the bicubic interpolation of the input and the ground truth. USes 3×3 convolution kernels, 10 residual blocks and RGB images (c = 3).

### Training Settings
Adversarial training along with an objective function consisting of texture matching loss and perceptual loss.

- **Perceptual Loss** : Difference between the SR output and target in the feature space of a differentiable function ![](https://latex.codecogs.com/gif.latex?\phi) (VGG-19 network).
- **Texture Matching Loss** : For matching the textures, *Gram matrix* is used as suggested in *Gatys et al., (2015)*[](https://arxiv.org/abs/1508.06576)
