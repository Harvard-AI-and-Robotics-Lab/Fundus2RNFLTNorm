import os
import numpy as np
import cv2
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    """Loads OCT fundus / RNFLT pairs from `.npz` files for Keras training."""

    def __init__(
        self,
        list_IDs,
        path,
        batch_size=8,
        dim=(256, 256),
        n_channels=3,
        shuffle=False,
    ):
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list(list_IDs)
        self.path = path
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self._data_generation(list_IDs_temp)

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _data_generation(self, list_IDs_temp):
        X1 = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        X2 = np.empty((self.batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((self.batch_size, *self.dim, 1), dtype=np.float32)

        for i, f in enumerate(list_IDs_temp):
            data = np.load(os.path.join(self.path, f))
            oct_fundus = cv2.resize(np.array(data['oct_fundus']), self.dim)
            rnflt = cv2.resize(np.array(data['rnflt']), self.dim)
            mask = np.ones(self.dim, dtype=np.float32)

            X1[i,] = np.transpose(np.array([oct_fundus, oct_fundus, oct_fundus]), (1, 2, 0))
            X2[i,] = np.transpose(np.array([mask, mask, mask]), (1, 2, 0))
            y[i,] = rnflt[:, :, np.newaxis]

        return (X1, X2), y


def prepare_octfundus_input(oct_fundus, dim=(256, 256)):
    """Convert a 2D OCT fundus image to the (image, mask) tensors expected by the model."""
    oct_fundus = cv2.resize(np.array(oct_fundus), dim)
    mask = np.ones(dim, dtype=np.float32)
    img = np.array([np.transpose(np.array([oct_fundus, oct_fundus, oct_fundus]), (1, 2, 0))])
    mask = np.array([np.transpose(np.array([mask, mask, mask]), (1, 2, 0))])
    return img, mask
