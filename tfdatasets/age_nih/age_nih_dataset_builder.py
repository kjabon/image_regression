"""
age dataset.
This file is not meant to be used out of the box, but is offered as
a starter example for those beginning with tfds.
"""
import dataclasses
import glob
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
import os,shutil

@dataclasses.dataclass
class BuilderConfig(tfds.core.BuilderConfig):
    img_size: int = 128
    ds_size:int = 110

class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for age dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'added dataset size',
    }

    def _info(self) -> tfds.core.DatasetInfo:

        """Returns the dataset metadata."""
        # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
        shape = self.builder_config.img_size

        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                # These are the features of your dataset like images, labels ...
                'image': tfds.features.Image(shape=(shape, shape, 1)),
                'label': tfds.features.Scalar(dtype=np.uint8)
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(my_dataset): Downloads the data and defines the splits
        # TODO(my_dataset): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            'train': self._generate_examples('train'),
            'val': self._generate_examples('val')
        }

    def _generate_examples(self, split):
        """Yields examples."""
        #TODO: need to manually download nih xray dataset from kaggle and fix the paths in this function
        fromCSV = pd.read_csv('../../xrayData/nihData/Data_Entry_2017_v2020.csv') # TODO download and fix paths
        count = 0
        splitCount = 0
        if split == 'train':
            toTest = set([1, 2, 3, 4, 5, 6, 7, 8, 9])
        elif split == 'val':
            toTest = set([0])
        else: toTest = [-1]
        for row in fromCSV.iterrows():

            if count not in toTest:
                count = (count + 1) % 10
                continue
            id = row[1].loc['Image Index']
            filename = id
            id = id.split('.')[0]
            age = int(row[1].loc['Patient Age'])
            path = '../../xrayData/nihData/' # TODO download and fix paths

            fullPath = path  + filename

            count = (count + 1) % 10
            splitCount+=1
            if not os.path.exists(fullPath):
                continue
            yield id, {
                'image': fullPath,
                'label': age
            }
