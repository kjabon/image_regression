# image_regression
Use jax/haiku to perform regression from x-ray data to age of the patient, and also implement RandAugment in jax (borrowing from others' code).

# blog post
See [this post](https://kjabon.github.io/blog/2023/ImageRegressionJax/) where I walk through the more interesting bits of code and training curves, working up to iteratively better results.

# dataset preparation
Note, you need to use tensorflow-datasets to package the [kaggle nih xray dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data) into something useable yourself.
Seek out the relevant documentation for creating your own tfds datasets.
[Here](https://www.tensorflow.org/datasets/add_dataset) is a good place to start.
For anyone interested in image processing, tfds is a worthwhile skill, and many learning resources already exist.

Example code for creating a tfds dataset can be found in:
``./tfdatasets/age_nih/age_nih_dataset_builder.py``
where it can be run with ``tfds build`` from the terminal in the enclosing directory, ``age_nih``, once the dataset has been downloaded.

# requirements
- jax ([install](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier) with gpu support)
- haiku
- optax
- tensorflow-datasets
- jmp (optional, you can comment the relevant bits out)
- numpy
- dm-tree
- abseil-py
- PIL
- GPUtil
- keras_cv (optional, can comment out the relevant bits in preprocess.py)
- tensorflow
- dm-pix
- imax
