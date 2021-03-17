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
    grads = np.load(os.path.join(prefix, usage, f"{file_name}_grads.npy"))
    depth_map = np.load(os.path.join(prefix, usage, f"{file_name}_depth_map.npy"))

    # reshape and calculate patch for loading into network
    x = np.moveaxis(grads, 0, -1)
    x = image.extract_patches_2d(x, patch_size)
    n_patchs, _, _, n_shifts = x.shape
    x = x.reshape(n_patchs, n_shifts * patch_size[0] * patch_size[1])
    print(x.shape)

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

#%%
def mse(gt, test):
    return np.mean((gt - test)**2)

#%%
n_shifts = 64
depth = "small"
# patch_size = (1, 1)
patch_size = (3, 3)
# patch_size = (5, 5)
network_layers = [256, 128, 64, 32]

# checkpoint_filepath= f"ckpts_{depth}_{patch_size[0]}/train"
# checkpoint_filepath = "ckpts_small_3_4_256_normal_drop/train"
checkpoint_filepath = "ckpts_all_3_4_256_normal_drop/train"
model_input_shape = n_shifts * patch_size[0] * patch_size[1]
model = build_mlp(model_input_shape, network_layers)
model.load_weights(checkpoint_filepath)

#%%
# refocus_depth, mlp_depth, depth_map_clip = evaluate(model, patch_size, "../data/rendered_processed", "val", "rosemary" ) # small val
file_name = "vinyl"
refocus_depth = get_refocus_map("../data/rendered_processed", "test", file_name)

#%%
file_name = "vinyl"
mlp_depth, depth_map_clip = evaluate(model, patch_size, "../data/rendered_processed", "test", file_name) # small test

#%%
plt.imshow(mlp_depth)
plt.colorbar()
#%%
file_name = "vinyl"
patch_size = (3, 3)

refocus_depth = get_refocus_map("../data/rendered_processed", "test", file_name)
depth_map_clip = get_gt_depth_map(patch_size, "../data/rendered_processed", "test", file_name)

disparty_map_mat = loadmat(f"{file_name}_disparty.mat")
defocus_corres_mat = loadmat(f"{file_name}_defocus_corres.mat")

center_view = disparty_map_mat['centerView']
disparty_map = disparty_map_mat['h_interp']
gt_disparty_map = disparty_map_mat['GT_disparity']

defocus_depth = defocus_corres_mat['defocus_depth']
defocus_corres_depth = defocus_corres_mat['depth_output']
corres_depth = defocus_corres_mat['corresp_depth']

grads = np.load(os.path.join("../data/rendered_processed", "test", f"{file_name}_grads.npy"))
defocus_stack = defocus_corres_mat['defocus_stack']

print(np.min(grads))
print(np.max(grads))

print(defocus_stack.shape)

print(np.min(defocus_stack))
print(np.max(defocus_stack))

print(np.min(refocus_depth))
print(np.max(refocus_depth))

print(np.min(defocus_depth))
print(np.max(defocus_depth))

print((refocus_depth + 1) - refocus_depth)


#%%
gt_depth_map_norm = normalize(depth_map_clip)
gt_depth_map_norm = (gt_depth_map_norm * 63).astype(np.uint8)

print(np.min(gt_depth_map_norm))
print(np.max(gt_depth_map_norm))

plt.imshow(gt_depth_map_norm)
plt.colorbar()

#%%
file_names = np.load(os.path.join("../data/rendered_processed", "train", "file_names.npy"))
print(file_names)

#%% crop


crop_x, crop_y = int((patch_size[0] - 1) / 2), int((patch_size[0] - 1) / 2)
gt_res_i, gt_res_j, c = center_view.shape

refocus_depth = refocus_depth[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]

center_view = center_view[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]
disparty_map = disparty_map[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]
gt_disparty_map = gt_disparty_map[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]
defocus_corres_depth = defocus_corres_depth[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]
defocus_depth = defocus_depth[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]
corres_depth = corres_depth[crop_x: gt_res_i-crop_x, crop_y: gt_res_j - crop_y]
#%%
# mlp_depth_norm = normalize(mlp_depth)

refocus_depth_norm = normalize(refocus_depth)
disparty_map_norm = normalize(disparty_map)
defocus_corres_depth_norm = normalize(defocus_corres_depth)
defocus_depth_norm = normalize(defocus_depth)
corres_depth_norm = normalize(corres_depth)

gt_depth_map_norm = normalize(depth_map_clip)

plt.imshow(np.concatenate((mlp_depth,
                           corres_depth_norm,
                           refocus_depth_norm,
                           defocus_depth_norm,
                           defocus_corres_depth_norm,
                           gt_depth_map_norm), axis=1))

print(mse(gt_depth_map_norm, mlp_depth),
    mse(gt_depth_map_norm, refocus_depth_norm),
    mse(gt_depth_map_norm, defocus_corres_depth_norm),
    mse(gt_depth_map_norm, corres_depth_norm))

#%%
plt.imshow(depth_map_clip_norm)
plt.colorbar()

#%%
plt.imshow(refocus_depth_norm)
plt.colorbar()

#%%
mlp_diff_norm = np.abs(depth_map_clip_norm - mlp_depth_norm)

mlp_mse = np.mean((depth_map_clip_norm - mlp_depth_norm)**2)
plt.imshow(mlp_diff_norm)
plt.colorbar()

#%%
refocus_diff_norm = np.abs(depth_map_clip_norm - refocus_depth_norm)

refocus_mse = np.mean((depth_map_clip_norm - refocus_depth_norm)**2)
plt.imshow(refocus_diff_norm)
plt.colorbar()

#%%
print(np.mean(mlp_diff_norm), np.mean(refocus_diff_norm))
print(np.mean(mlp_mse), np.mean(refocus_mse))

#%%
plt.imshow(y_gt_clip)
plt.colorbar()

#%%
plt.imshow(np.concatenate((y_test_clip, y_gt_clip), axis=1))
plt.colorbar()

#%%
diff = np.abs(y_test - y_gt_clip)
plt.imshow(diff)
plt.colorbar()

print(np.mean(diff))

# %%
percentage = diff / y_gt_clip
plt.imshow(percentage)
plt.colorbar()
