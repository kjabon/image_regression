# image_regression
Use jax/haiku to perform regression from x-ray data to age of the patient

# dataset preparation
Note, you need to use tensorflow-datasets to package the [kaggle nih xray dataset](https://www.kaggle.com/datasets/nih-chest-xrays/data) into something useable yourself.
Seek out the relevant documentation for creating your own tfds datasets.
[Here](https://www.tensorflow.org/datasets/add_dataset) is a good place to start.
For anyone interested in image processing, tfds is a worthwhile skill, and many learning resources already exist.

Example code for creating a tfds dataset can be found in:
``./tfdatasets/age_nih/age_nih_dataset_builder.py``
where it can be run with ``tfds build`` from the terminal in the enclosing directory, ``age_nih``, once the dataset has been downloaded.
