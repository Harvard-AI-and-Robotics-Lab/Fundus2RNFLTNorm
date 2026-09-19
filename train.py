import argparse
import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

from tensorflow.keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from models.reconstruct import PCModel
from utils.data_process import DataGenerator
from utils.map_handler import gen_cmap


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train OCT fundus to RNFLT map prediction'
    )
    parser.add_argument('--model-name', default='octfundus_to_rnflt_new2')
    parser.add_argument('--data-path', default='dataset/', help='folder of .npz samples')
    parser.add_argument(
        '--meta-csv',
        default='dataset/metatable.csv',
        help='CSV with columns filename and train (1=train, 0=held-out)',
    )
    parser.add_argument('--checkpoint-dir', default='checkpoint/')
    parser.add_argument('--log-dir', default='imgs/')
    parser.add_argument('--img-rows', type=int, default=256)
    parser.add_argument('--img-cols', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument(
        '--val-size',
        type=int,
        default=99,
        help='number of held-out samples used for validation',
    )
    parser.add_argument('--gpu', default='0')
    parser.add_argument('-f', default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def configure_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        return
    try:
        for device in gpus:
            tf.config.experimental.set_memory_growth(device, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPUs')
    except RuntimeError as exc:
        print(exc)


def load_splits(meta_csv, val_size):
    meta = pd.read_csv(meta_csv)
    if 'filename' not in meta.columns or 'train' not in meta.columns:
        raise ValueError('meta CSV must contain columns: filename, train')

    train_paths = list(meta[meta['train'] == 1]['filename'])
    held_out = list(meta[meta['train'] == 0]['filename'])
    if len(train_paths) == 0:
        raise ValueError('No training samples found (train==1).')
    if len(held_out) == 0:
        raise ValueError('No held-out samples found (train==0).')

    val_size = min(val_size, len(held_out))
    val_paths = held_out[:val_size]
    test_paths = held_out[val_size:]
    print(
        'n_train={}, n_val={}, n_test={}'.format(
            len(train_paths), len(val_paths), len(test_paths)
        )
    )
    return train_paths, val_paths, test_paths


def build_plot_callback(model, generator, model_name, log_dir):
    color = gen_cmap()
    X_vis, y_vis = generator[0]

    def plot_callback(_model):
        pred_img = _model.predict(X_vis)
        os.makedirs(log_dir, exist_ok=True)
        for i in range(len(y_vis)):
            _, axes = plt.subplots(1, 3, figsize=(20, 5))
            axes[0].imshow(X_vis[0][i, :, :, 0], cmap='gray')
            axes[1].imshow(pred_img[i, :, :, 0], cmap=color, vmin=0, vmax=350)
            axes[2].imshow(y_vis[i, :, :, 0], cmap=color, vmin=0, vmax=350)
            axes[0].set_title('OCT fundus')
            axes[1].set_title('Predicted RNFLT')
            axes[2].set_title('Ground-truth RNFLT')
            axes[0].axis('off')
            axes[1].axis('off')
            axes[2].axis('off')
            plt.savefig(os.path.join(log_dir, '{}_img_{}.png'.format(model_name, i)))
            plt.close()

    return plot_callback


def main():
    args = parse_args()
    configure_gpu(args.gpu)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    train_paths, val_paths, _ = load_splits(args.meta_csv, args.val_size)
    dim = (args.img_rows, args.img_cols)

    train_generator = DataGenerator(
        train_paths,
        args.data_path,
        batch_size=args.batch_size,
        dim=dim,
        shuffle=True,
    )
    val_generator = DataGenerator(
        val_paths,
        args.data_path,
        batch_size=args.batch_size,
        dim=dim,
        shuffle=False,
    )

    model = PCModel(args.img_rows, args.img_cols).build_pconv_unet()
    try:
        optimizer = Adam(lr=args.lr)
    except TypeError:
        optimizer = Adam(learning_rate=args.lr)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    plot_callback = build_plot_callback(
        model, val_generator, args.model_name, args.log_dir
    )
    start_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        validation_data=val_generator,
        validation_steps=len(val_generator),
        epochs=args.epochs,
        verbose=2,
        callbacks=[
            TensorBoard(log_dir=args.checkpoint_dir, write_graph=False),
            ModelCheckpoint(
                os.path.join(
                    args.checkpoint_dir,
                    args.model_name + '.{epoch:02d}-{loss:.6f}.h5',
                ),
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
            ),
            LambdaCallback(on_epoch_end=lambda epoch, logs: plot_callback(model)),
        ],
    )

    end_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print(start_time, end_time)

    final_path = os.path.join(args.checkpoint_dir, args.model_name + '.final.h5')
    model.save_weights(final_path)
    np.save(os.path.join(args.checkpoint_dir, args.model_name + '_train_history.npy'), history.history)
    print('Saved final weights to', final_path)
    print('Training finished!')


if __name__ == '__main__':
    main()
