Accelerated Training
====================

NVIDIA Merlin accelerates the recommendation pipeline from end to end. Applying deep learning models to the recommendation pipeline has unique challenges in comparison to other domains, such as computer vision and natural language processing. With the constant additions of new items and users along with their evolving content desires, the deep learning model needs to be updated regularly. To achieve this, NVIDIA Merlin offers three accelerated dataloaders.

Recommendation system datasets can be terra-bytes in size with billions of examples but each example is represented by only a few bytes. For example, the Criteo CTR dataset, which is the largest publicly available dataset, has 1.3TB with 4 billion samples. The model architectures typically have large embedding tables with hundreds of millions of users and items that do not fit on a single GPU.

The base dataloader in PYTorch and TensorFlow randomly sample each item from the dataset, which is very slow. The window dataloader in TensorFlow isn’t any faster.  In our experiments, we were able to speed up existing TensorFlow pipelines by 9 times with our highly optimized dataloaders. HugeCTR, our dedicated deep learning framework for recommender systems can achieve speed-ups up to 13 times. In addition, HugeCTR supports model parallel scaling for embedding tables that require a lot of memory. HugeCTR can distribute an embedding table over multiple GPUs or multiple nodes. 

For additional information, see our latest `blog post
<https://medium.com/nvidia-merlin/why-isnt-your-recommender-system-training-faster-on-gpu-and-what-can-you-do-about-it-6cb44a711ad4>`_.

.. toctree::
   :maxdepth: 2

   Tensorflow <tensorflow>
   PyTorch <pytorch>
   HugeCTR <hugectr>
