
<p align="center">
  <img src="https://github.com/xiaoxingxingkz/BrainTransGAN/blob/main/img/subject_s.gif" width="400">
</p>

# BrainStatTrans-GAN: Decoding individual differences of brain dynamics in Alzheimer’s Disease by transferring generative adversarial networks (Updating)
## Abstract

Deep learning has been widely investigated in brain image computing and analysis for disease diagnosis. Most of existing methods build the deep learning models to learn the features from brain images, followed by group analysis to classify diseases. It is still challenging to model the individual brain dynamics in disease for interpretable imaging and precision medicine. In this paper, we propose a generative model based on brain status transferring generative adversarial network (BrainStatTrans-GAN) to decode the individual differences of brain dynamics in Alzheimer’s Disease. The BrainStatTrans-GAN consists of 3 components which are generator, discriminator, and status discriminator. First, a generative adversarial network with generator and discriminator is built to generate the heathy brain images. Then, a status discriminator is added to the generator to produce the heathy brain images from the disease. Finally, the differences between the generated and real images are computed to decode the brain dynamics, which can be used for disease diagnosis and interpretation of brain changes. Compared to the existing group analysis, the proposed method can model the individualized brain dynamics in Alzheimer’s Disease which can facilitate the disease diagnosis and interpretation. Interpretable experiments on three datasets with 1739 subjects demonstrated that our BrainTransGAN can be used as a tool with superior properties for reconstructing healthy images and exploring whole brain dynamics. 

## Overview
<p align="center">
  <img src="https://github.com/xiaoxingxingkz/BrainTransGAN/blob/main/img/F1.png" width="700">
</p>

## Prerequisites

* Linux
* Python3.7
* MindSpore 2.0.0-alpha
* CPU or NVIDIA GPU + CUDA CuDNN

## Implementation details
Our BrainStatTrans-GAN was implemented with python 3.7.4 in the Pytorch framework. All experiments were finished on the GPU of NVIDIA GeForce RTX3090 with Ubuntu system. We used ANDI-1 dataset as the training set and stacked the ADNI-2, AIBL, and OASIS datasets as the testing set. 


At phase 1, we only input the healthy images (i.e., NC images). Adam optimizer was used with the batch size of 8. The learning rates of the generator and discriminator were set to 1e-4 and 4e-4, respectively. For the generator, we introduced a set of coefficients to balance the training process. 


At phase 2, we input the images in disease (AD or MCI) and normal control (NC). Adam optimizer was used with the batch size of 8. The generator inherited the final weights from phase 1 with the learning rate as 1e-4. The learning rate of the status discriminator was set to 1e-3. Moreover, we ran one epoch of phase 1 after running 10 epochs of phase 2 to hold the normal regions in the brain unchanged. 150 epochs for phase 2 and another 15 epochs for phase 1 are needed. 
