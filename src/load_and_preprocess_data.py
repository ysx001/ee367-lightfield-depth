import numpy as np
import skimage.io as io
import os
from tqdm import tqdm
from scipy.interpolate import interp2d
from itertools import product

import file_io

def load_data_folder(data_dir):
    directories = []
    for (dirpath, _, _) in os.walk(data_dir):
        if dirpath == data_dir:
            continue
        directories.append(dirpath)
    return directories

def calcualte_refocus_shifts(depth_map, depth_res=64):
    depth_max = np.max(depth_map)
    depth_min = np.min(depth_map)
    depth_step = (depth_max - depth_min) / depth_res
    refocus_shifts = np.arange(depth_min, depth_max, depth_step)
    print(f"Min Depth: {depth_min}, Max Depth: {depth_max}, step: {depth_step}")
    print(refocus_shifts[0], refocus_shifts[-1])
    return refocus_shifts

def load_rendered_light_field(data_folder):
    params = file_io.read_parameters(data_folder)
    light_field = file_io.read_lightfield(data_folder)
    depth_map_lowres = file_io.read_depth(data_folder, highres=False)
    # make sure the dimensions match with lytro images
    light_field = np.moveaxis(light_field, [0, 1], [2, 3])
    width, depth, microlens_size, microlens_size, color_channel = light_field.shape
    print(light_field.shape)
    X, Y = np.array(range(depth)), np.array(range(width))
    interp_fns = {}
    for ky in range(microlens_size):
        interp_fns[ky] = {}
        for kx in range(microlens_size):
            interp_fns[ky][kx] = []
            lf = light_field[:, :, ky, kx, :]
            for c in range(color_channel):
                interp_fns[ky][kx].append(interp2d(X, Y, lf[..., c]))
    return light_field, depth_map_lowres, interp_fns

def load_lytro_light_field(image_path, microlens_size, depth_map_path=None):
    LF_microlens = io.imread(image_path).astype(np.float32)/255
    print(LF_microlens.shape)

    light_field_resolution = (int(LF_microlens.shape[0]/microlens_size),
                            int(LF_microlens.shape[1]/microlens_size),
                            int(microlens_size),
                            int(microlens_size),
                            int(LF_microlens.shape[2]))
    light_field = np.zeros(light_field_resolution)
    print(light_field_resolution)

    # Extract light field and interpolation functions
    X, Y = np.array(range(light_field_resolution[1])), np.array(range(light_field_resolution[0]))
    interp_fns = {}
    print("Extracting subimages...")
    for ky in range(microlens_size):
        interp_fns[ky] = {}
        for kx in range(microlens_size):
            interp_fns[ky][kx] = []
            lf = LF_microlens[ky::microlens_size, kx::microlens_size, :]
            light_field[:, :, ky, kx, :] = lf
            for c in range(light_field.shape[4]):
                interp_fns[ky][kx].append(interp2d(X, Y, lf[..., c]))

    # Loading Depth Map
    if depth_map_path is not None:
        depth_map = io.imread(depth_map_path).astype(np.float32)
        print(depth_map.shape)
        return light_field, depth_map, interp_fns
    return light_field, None, interp_fns

def calculate_refocus(light_field, interp_fns, refocus_shifts):
    X, Y = np.array(range(light_field.shape[1])), np.array(range(light_field.shape[0]))
    microlens_size = light_field.shape[2]
    grads = np.zeros((len(refocus_shifts), light_field.shape[0], light_field.shape[1]))
    focus_stack = np.zeros((len(refocus_shifts), light_field.shape[0], light_field.shape[1], light_field.shape[4]))
    print("Shifting images")
    for g, refocusShift in enumerate(refocus_shifts):
        stack = np.zeros((microlens_size ** 2, light_field.shape[0], light_field.shape[1], light_field.shape[4]))
        for i, (ky, kx) in tqdm(enumerate(product(range(microlens_size), range(microlens_size)))):
            maxU = (microlens_size - 1)/2
            maxV = (microlens_size - 1)/2
            # Get u, v coordinates (normalized to [-1,1])
            u = (ky - maxU)/maxU
            v = (kx - maxV)/maxV
            # print(u, v)

            # Shift each channel
            shifted = np.zeros((light_field.shape[0], light_field.shape[1], light_field.shape[4]))
            for c in range(light_field.shape[4]):
                shifted[..., c] = interp_fns[ky][kx][c](X + v*refocusShift, Y + u*refocusShift)
            stack[i, ...] = shifted
        refocused = np.sum(stack, axis=0)/(microlens_size ** 2)
        focus_stack[g, ...] = refocused
        # Compute gradients
        Dx = np.diff(refocused, axis=1, append=0)
        Dy = np.diff(refocused, axis=0, append=0)
        grads[g, ...] = np.sum(np.sqrt(Dx ** 2 + Dy ** 2), axis=-1)
    return focus_stack, grads

def load_data(prefix, usage="train"):
    data_directories = load_data_folder(os.path.join(prefix, usage))
    file_names = []
    file_names_file_name = "file_names.npy"
    result_path = f"{prefix}_processed"
    for data_directory in tqdm(data_directories):
        file_name = data_directory.split('/')[-1]
        file_names.append(file_name)
        print(f'Loading Light Field Data {file_name}')
        # load light field data
        light_field, depth_map, interp_fns = load_rendered_light_field(data_directory)
        # decide what refocus shifts to apply
        refocus_shifts = calcualte_refocus_shifts(depth_map)
        print(len(refocus_shifts))
        # Calculate refocus stack and gradients
        focus_stack, grads = calculate_refocus(light_field, interp_fns, refocus_shifts)
        # save results
        grads_file_name = f"{file_name}_grads.npy"
        focus_stack_file_name = f"{file_name}_focus_stack.npy"
        depth_map_file_name = f"{file_name}_depth_map.npy"
        np.save(os.path.join(result_path, usage, depth_map_file_name), depth_map)
        np.save(os.path.join(result_path, usage, grads_file_name), grads)
        np.save(os.path.join(result_path, usage, focus_stack_file_name), focus_stack)
    np.save(os.path.join(result_path, usage, file_names_file_name), np.array(file_names))

# load_data("../data/rendered", usage="train")
# load_data("../data/rendered", usage="val")
# load_data("../data/rendered", usage="test")

