import argparse
import os

import cv2
import numpy as np
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

from tensorflow.keras.optimizers import Adam

from models.reconstruct import PCModel
from utils.data_process import prepare_octfundus_input
from utils.map_handler import plot_2dmap


def parse_args():
    parser = argparse.ArgumentParser(
        description='Infer RNFLT maps from OCT fundus images'
    )
    parser.add_argument(
        '--weights',
        default='checkpoint/octfundus_to_rnflt_new2.final.h5',
        help='path to pretrained .h5 weights',
    )
    parser.add_argument('--input', required=True, help='.npz sample or a 2D OCT fundus .npy')
    parser.add_argument('--output', default=None, help='optional .npy path to save the prediction')
    parser.add_argument('--img-rows', type=int, default=256)
    parser.add_argument('--img-cols', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--show', action='store_true', help='plot the predicted RNFLT map')
    return parser.parse_args()


def load_oct_fundus(path):
    if path.endswith('.npz'):
        data = np.load(path)
        if 'oct_fundus' not in data.files:
            raise ValueError('NPZ file must contain an oct_fundus array.')
        return np.array(data['oct_fundus']), data
    return np.load(path), None


def build_model(img_rows, img_cols, lr, weights):
    model = PCModel(img_rows, img_cols).build_pconv_unet(train_bn=False)
    try:
        optimizer = Adam(lr=lr)
    except TypeError:
        optimizer = Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    model.load_weights(weights)
    return model


def main():
    args = parse_args()
    oct_fundus, data = load_oct_fundus(args.input)
    model = build_model(args.img_rows, args.img_cols, args.lr, args.weights)

    img, mask = prepare_octfundus_input(oct_fundus, dim=(args.img_rows, args.img_cols))
    pred_rnflt = model.predict((img, mask))[0][:, :, 0]

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        np.save(args.output, pred_rnflt)
        print('Saved predicted RNFLT map to', args.output)

    if args.show:
        plot_2dmap(pred_rnflt, title='Predicted RNFLT')
        if data is not None and 'rnflt' in data.files:
            gt = cv2.resize(np.array(data['rnflt']), (args.img_cols, args.img_rows))
            plot_2dmap(gt, title='Ground-truth RNFLT')

    print('Predicted RNFLT shape:', pred_rnflt.shape)
    print('Mean RNFLT: {:.2f} um'.format(float(np.mean(pred_rnflt))))


if __name__ == '__main__':
    main()
