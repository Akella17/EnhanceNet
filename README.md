## EnhanceNet

This is an implementation of the paper ["EnhanceNet: Single Image Super-Resolution Through Automated Texture Synthesis"](https://arxiv.org/abs/1612.07919) by *M. Sajjadi et al., (2017)*.

### Model Architecture

![](https://github.com/Akella17/EnhanceNet/raw/master/data/enhancenet_architecture.png)

Fully convolutional network architecture for 4x super-resolution which only learns the residual between the bicubic interpolation of the input and the ground truth. USes 3×3 convolution kernels, 10 residual blocks and RGB images (c = 3).

### Training Objective
Perceptual Loss and  automated texture synthesis
