#%%
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import hsv2rgb
from pathlib import Path
from tqdm import tqdm
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.feature_extraction import image
import os
import argparse

import file_io

def img_to_uint8(img):
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)

def plot_all_in_focus(focus_stack, max_ind):
    all_in_focus = np.zeros((light_field.shape[0], light_field.shape[1], light_field.shape[4]))
    for y in range(light_field.shape[0]):
        for x in range(light_field.shape[1]):
            all_in_focus[y, x, :] = focus_stack[max_ind[y, x], y, x, :]
    io.imsave('all_in_focus.png', img_to_uint8(all_in_focus ** (1/2.2)))

def build_mlp(input_shape, layers, leaky_alpha=0.1, dropout_rate=0.1):
    # Create model architecture
    model = Sequential()
    model.add(Dense(units=input_shape, input_shape=(input_shape, ), activation='relu'))
    for layer in layers:
        model.add(Dense(units=layer))
        model.add(LeakyReLU(alpha=leaky_alpha)) # activation between fully connected layers
        model.add(Dropout(rate=dropout_rate)) # prevent overfitting
    model.add(Dense(units=1, activation='relu'))
    return model

LARGE = ['tomb', 'platonic', 'sideboard', 'dishes', 'town', 'dots', 'pyramids', 'tower'] # min depth > 10
MEDIUM = ['antinous', 'stripes', 'dino', 'pens', 'greek', 'medieval2', 'backgammon', 'museum'] # min depth > 4 < 10
SMALL = ['boxes', 'kitchen', 'table', 'boardgames', 'cotton', 'pillows', 'rosemary', 'vinyl'] # min depth > 4 < 10
def get_data(prefix, usage, sample_data_size=None, patch_size=(1, 1), depth="small", normalize=True):
    file_names = np.load(os.path.join(prefix, usage, "file_names.npy"))
    idx = 0
    for file_name in file_names:
        if depth == "small" and file_name not in SMALL:
            continue
        elif depth == "medium" and file_name not in MEDIUM:
            continue
        elif depth == "large" and file_name not in LARGE:
            continue
        print(file_name)
        x = np.load(os.path.join(prefix, usage, f"{file_name}_grads.npy"))
        x = np.moveaxis(x, 0, -1)
        x = image.extract_patches_2d(x, patch_size)
        n_patchs, _, _, n_shifts = x.shape
        x = x.reshape(n_patchs, n_shifts * patch_size[0] * patch_size[1])
 

        y_gt = np.load(os.path.join(prefix, usage, f"{file_name}_depth_map.npy"))
        gt_res_i, gt_res_j = y_gt.shape
        clip_x, clip_y = int((patch_size[0] - 1) / 2), int((patch_size[0] - 1) / 2)
        y_gt_clip = y_gt[clip_x: gt_res_i-clip_x, clip_y: gt_res_j - clip_y]
        res_i, res_j = y_gt_clip.shape
        y = y_gt_clip.reshape(res_i * res_j)
        if normalize:
            y = (y - np.min(y)) / np.ptp(y)

        if patch_size[0] == 3:
            sample_data_size = 100000
        if patch_size[0] == 5:
            sample_data_size = 50000
        if sample_data_size is not None:
            sample_idx = np.random.choice(n_patchs, sample_data_size, replace=False)
            x = x[sample_idx, :]
            y = y[sample_idx]

        X = x if idx == 0 else np.concatenate((X, x), axis=0)
        print(X.shape)

        Y = y if idx == 0 else np.concatenate((Y, y), axis=0)
        print(Y.shape)
        idx += 1
    return X, Y

def train(patch_size, depth="small", n_shifts=64, normalize=True,
          network_layers=[64, 32, 16], lr=0.0001,
          leaky_alpha=0.1, dropout_rate=0.1, 
          batch_size=512, data_prefix="../data/rendered_processed"):
    train_x, train_y = get_data(data_prefix, "train", patch_size=patch_size, depth=depth, normalize=normalize)
    val_x, val_y = get_data(data_prefix, "val", patch_size=patch_size, depth=depth, normalize=normalize)

    checkpoint_filepath= f"ckpts_{depth}_{patch_size[0]}_{len(network_layers)}_{network_layers[0]}_normal_drop/train"
    
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_mean_squared_error',
        save_weights_only=True,
        save_best_only=True,
        mode='min')

    early_stop_callback = EarlyStopping(monitor='loss', patience=10)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    model_input_shape = n_shifts * patch_size[0] * patch_size[1]
    model = build_mlp(model_input_shape, network_layers, leaky_alpha=leaky_alpha, dropout_rate=dropout_rate)

    opt = Adam(learning_rate=lr)
    model.compile(loss='mean_absolute_error', optimizer=opt, metrics=['mean_squared_error'])
    history = model.fit(train_x, train_y,
                        validation_data=(val_x, val_y),
                        epochs=10000, batch_size=batch_size,
                        callbacks=[early_stop_callback, model_checkpoint_callback, reduce_lr])
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size',
                        help='Patch size',
                        type=int)
    parser.add_argument('--depth',
                        help='small medium or lange',
                        type=str)
    parser.add_argument('--lr',
                        help='learning rate',
                        type=float)
    parser.add_argument('--network_layers',
                        help='Network layers comma delimited',
                        type=str)
    args = parser.parse_args()

    # %%
    n_shifts = 64
    # patch_size = (3, 3)
    patch_size = (args.patch_size, args.patch_size)
    network_layers = [int(i) for i in args.network_layers.split(',')]
    # network_layers=[256, 128, 64, 32]
    # network_layers=[256, 128]

    #%%
    model = train(data_prefix="../data/rendered_processed",
                  patch_size=patch_size,
                  n_shifts=n_shifts,
                  depth=args.depth,
                  batch_size=512,
                  lr=args.lr,
                  network_layers=network_layers)