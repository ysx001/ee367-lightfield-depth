#%%
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.color import hsv2rgb
from pathlib import Path
from tqdm import tqdm
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from sklearn.feature_extraction import image
import os
import argparse
from scipy.io import loadmat

from train import build_mlp
from utils import calcualte_refocus_shifts


def get_refocus_map(prefix, usage, file_name):
    grads = np.load(os.path.join(prefix, usage, f"{file_name}_grads.npy"))
    max_ind = np.argmax(grads, axis=0)
    return max_ind

def get_gt_depth_map(patch_size, prefix, usage, file_name):
    depth_map = np.load(os.path.join(prefix, usage, f"{file_name}_depth_map.npy"))
    clip_x, clip_y = int((patch_size[0] - 1) / 2), int((patch_size[0] - 1) / 2)
    gt_res_i, gt_res_j = depth_map.shape
    depth_map_clip = depth_map[clip_x: gt_res_i-clip_x, clip_y: gt_res_j - clip_y]
    return depth_map_clip


def evaluate(model, patch_size, prefix, usage, file_name):
    # load data
    # grads = np.load(os.path.join(prefix, usage, f"{file_name}_grads.npy"))
    depth_map = np.load(os.path.join(prefix, usage, f"{file_name}_depth_map.npy"))

    x_grad = np.load(os.path.join(prefix, usage, f"{file_name}_grads.npy"))
    x_grad = np.moveaxis(x_grad, 0, -1)
    x_grad = image.extract_patches_2d(x_grad, patch_size)
    n_patchs, _, _, n_shifts = x_grad.shape
    x_grad = x_grad.reshape(n_patchs, n_shifts * patch_size[0] * patch_size[1])

    x_defocus = np.load(os.path.join(prefix, usage, f"{file_name}_defocus_stack.npy"))
    x_defocus_max_ind = np.argmax(x_defocus, axis=2)
    x_defocus_max_ind = (x_defocus_max_ind - np.min(x_defocus_max_ind)) / np.ptp(x_defocus_max_ind)
    x_defocus_max_ind = image.extract_patches_2d(x_defocus_max_ind, patch_size)
    x_defocus_max_ind = x_defocus_max_ind.reshape(n_patchs, patch_size[0] * patch_size[1])
    x_defocus = image.extract_patches_2d(x_defocus, patch_size)
    n_patchs, _, _, n_shifts = x_defocus.shape
    x_defocus = x_defocus.reshape(n_patchs, n_shifts * patch_size[0] * patch_size[1])
    

    x_corres = np.load(os.path.join(prefix, usage, f"{file_name}_correspondence_stack.npy"))
    x_corres_min_ind = np.argmin(x_corres, axis=2)
    x_corres_min_ind = (x_corres_min_ind - np.min(x_corres_min_ind)) / np.ptp(x_corres_min_ind)
    x_corres_min_ind = image.extract_patches_2d(x_corres_min_ind, patch_size)
    x_corres_min_ind = x_corres_min_ind.reshape(n_patchs, patch_size[0] * patch_size[1])
    # x_corres = np.moveaxis(x_corres, 0, -1)
    x_corres = image.extract_patches_2d(x_corres, patch_size)
    n_patchs, _, _, n_shifts = x_corres.shape
    x_corres = x_corres.reshape(n_patchs, n_shifts * patch_size[0] * patch_size[1])

    x = np.concatenate((x_grad, x_defocus, x_corres, x_defocus_max_ind, x_corres_min_ind), axis=1)

    # clip the boundaries of y according to the patch size
    clip_x, clip_y = int((patch_size[0] - 1) / 2), int((patch_size[0] - 1) / 2)
    gt_res_i, gt_res_j = depth_map.shape
    depth_map_clip = depth_map[clip_x: gt_res_i-clip_x, clip_y: gt_res_j - clip_y]

    mlp_depth = model.predict(x).reshape(depth_map_clip.shape)
    return mlp_depth, depth_map_clip

def get_refocus_depth_map(grads):
    max_ind = np.argmax(grads, axis=0)
    return max_ind

def hsv_depth(refocus_shifts, max_ind, grads):
    # HSV-based depth
    H = np.array(refocus_shifts)[max_ind]
    H = (H - np.min(H))/(np.max(H) - np.min(H))
    S = np.ones_like(max_ind)
    V = np.max(grads, axis=0)
    V /= np.max(V)
    hsv = np.stack((H, S, V), axis=-1)
    depth = hsv2rgb(hsv)
    depth = np.clip(depth, 0., 1.)
    return depth

def normalize(img):
    return (img - np.min(img)) / np.ptp(img)

def mse(gt, test):
    return np.mean((gt - test)**2)

def visualize_results(prefix, file_name, usage, output_path, model, n_shifts, patch_size, network_layers):
    mlp_depth, depth_map_clip = evaluate(model, patch_size, prefix, usage, file_name)

    # load other method results for comparison
    refocus_depth = get_refocus_map(prefix, usage, file_name)
    defocus_corres_mat = loadmat(os.path.join(prefix, usage, "defocus_correspondence_results", f"{file_name}_defocus_corres.mat"))
    defocus_depth = defocus_corres_mat['defocus_depth']
    defocus_corres_depth = defocus_corres_mat['depth_output']
    corres_depth = defocus_corres_mat['corresp_depth']
    defocus_stack = defocus_corres_mat['defocus_stack']
    center_view = defocus_corres_mat['centerView']

    #%% crop
    crop_x, crop_y = int((patch_size[0] - 1) / 2), int((patch_size[0] - 1) / 2)
    gt_res_i, gt_res_j = defocus_depth.shape

    refocus_depth = refocus_depth[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]
    defocus_depth = defocus_depth[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]
    corres_depth = corres_depth[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]
    defocus_corres_depth = defocus_corres_depth[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]
    center_view = center_view[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]

    mlp_depth_clip = np.clip(mlp_depth, 0.0, 1.0)

    refocus_depth_norm = normalize(refocus_depth)
    defocus_depth_norm = normalize(defocus_depth)
    corres_depth_norm = normalize(corres_depth)
    defocus_corres_depth_norm = normalize(defocus_corres_depth)
    gt_depth_map_norm = normalize(depth_map_clip)

    plt.imshow(center_view)
    plt.axis("off")
    plt.savefig(os.path.join(output_path, f"{file_name}_center_view.png"))
    plt.close("all")

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    plt.figure(figsize=(120,30))
    ax = plt.gca()
    im = ax.imshow(np.concatenate((
                            refocus_depth_norm,
                            corres_depth_norm,
                            defocus_depth_norm,
                            defocus_corres_depth_norm,
                            mlp_depth_clip,
                            gt_depth_map_norm), axis=1))
    # plt.imshow()
    plt.axis("off")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="1%", pad=0.5)
    plt.colorbar(im, cax=cax)
    plt.savefig(os.path.join(output_path, f"{file_name}_depth_map_comparison.png"))
    plt.close("all")

    print(f"{file_name} refocus mse={mse(gt_depth_map_norm, refocus_depth_norm)}, \
            correspondence mse={mse(gt_depth_map_norm, corres_depth_norm)}, \
            defocus mse={mse(gt_depth_map_norm, defocus_depth_norm)}, \
            defocus corres mse={mse(gt_depth_map_norm, defocus_corres_depth_norm)}, \
            mlp mse={mse(gt_depth_map_norm, mlp_depth_clip)}")

if __name__ == '__main__':
    n_shifts = 64 # best
    # patch_size = (1, 1)
    patch_size = (3, 3) # best
    # patch_size = (5, 5)
    network_layers = [256, 128, 64, 32] # best
    # network_layers = [128, 64, 32]

    #%%
    # checkpoint_filepath = "checkpoints/ckpts_small_3_4_256_normal_drop/train"
    # checkpoint_filepath = "checkpoints/ckpts_all_3_4_256_normal_drop/train"
    # checkpoint_filepath = "checkpoints/ckpts_small_1_4_256_normal_0.2_defocus_corres/train"
    # checkpoint_filepath = "checkpoints/ckpts_small_1_3_128_normal_0.2_defocus_corres/train"
    # checkpoint_filepath = "checkpoints/ckpts_all_1_4_256_normal_0.2_defocus_corres/train" 
    checkpoint_filepath = "checkpoints/ckpts_all_3_4_256_normal_0.2_defocus_corres/train" # best
    model_input_shape = n_shifts * patch_size[0] * patch_size[1] * 3 + (2 * patch_size[0] * patch_size[1])
    model = build_mlp(model_input_shape, network_layers)
    model.load_weights(checkpoint_filepath)

    usage="test"
    file_names = ["vinyl", "tower", "museum"]

    output_path="visual_output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file_name in file_names:
        visualize_results("../data/rendered_processed", file_name, usage, output_path, model, n_shifts, patch_size, network_layers)