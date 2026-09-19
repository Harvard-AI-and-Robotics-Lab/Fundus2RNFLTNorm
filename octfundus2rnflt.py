"""Public API for OCT-fundus-to-RNFLT inference."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

for _name in list(sys.modules):
    if _name == 'models' or _name.startswith('models.') or _name == 'utils' or _name.startswith('utils.'):
        del sys.modules[_name]

# Select the device before TensorFlow initializes CUDA. Set CUDA_VISIBLE_DEVICES
# to a free GPU, or to '' to run on CPU.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Without memory growth TF reserves most of the GPU up front, which makes cuBLAS
# fail to initialize on a shared machine.
for _gpu in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError as _exc:
        print(_exc)

from models.reconstruct import PCModel
from utils.data_process import prepare_octfundus_input
from utils.map_handler import gen_cmap, plot_2dmap

DEFAULT_WEIGHTS = str(ROOT / 'checkpoint' / 'octfundus_to_rnflt_new2.final.h5')
SAMPLE_IMAGE = str(ROOT / 'samples' / 'sample_oct_fundus.png')


def construct_model(img_rows=256, img_cols=256):
    return PCModel(img_rows=img_rows, img_cols=img_cols)


def load_sample(path=None, dim=(256, 256)):
    """Load an OCT fundus image and return (oct_fundus, img, mask) for prediction."""
    if path is None:
        path = SAMPLE_IMAGE
    oct_fundus = cv2.resize(cv2.imread(path, 0), dim)
    img = np.array([np.transpose(np.array([oct_fundus, oct_fundus, oct_fundus]), (1, 2, 0))])
    mask = np.ones_like(img)
    return oct_fundus, img, mask


__all__ = [
    'PCModel',
    'construct_model',
    'load_sample',
    'prepare_octfundus_input',
    'plot_2dmap',
    'gen_cmap',
    'DEFAULT_WEIGHTS',
    'SAMPLE_IMAGE',
    'ROOT',
    'cv2',
    'np',
    'plt',
]
