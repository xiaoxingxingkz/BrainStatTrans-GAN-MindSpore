


# Multimodal Transformer Network for Incomplete Image Generation and Diagnosis of Alzheimer’s Disease

## Abstract

Benefiting from complementary information, multimodal brain imaging analysis has distinct advantages over single-modal methods for diagnosis of neurodegenerative diseases such as Alzheimer’s disease. However, multi-modal brain images are often incomplete with missing data in clinical prac-tice due to various issues such as motion, medical costs, and scanner availa-bility. Most existing methods attempted to build machine learning models to directly estimate the missing images. However, since brain images are of high dimension, accurate and efficient estimation of missing data is quite challenging, and not all voxels in the brain images are associated with the disease. In this paper, we propose a multimodal feature-based transformer to impute multimodal brain features with missing data for diagnosis of neuro-degenerative disease. The proposed method consists of a feature regression subnetwork and a multimodal fusion subnetwork based on transformer, for completion of the features of missing data and also multimodal diagnosis of disease. Different from previous methods for generation of missing images, our method imputes high-level and disease-related features for multimodal classification. Experiments on ADNI database with 1,364 subjects show bet-ter performance of our method over the state-of-the-art methods in disease diagnosis with missing multimodal data.. 


## Prerequisites

* Linux
* Python3.7
* MindSpore 2.0.0-alpha
* CPU or NVIDIA GPU + CUDA CuDNN


