import numpy as np
from tqdm import tqdm
from itertools import product

def calcualte_refocus_shifts(depth_map, depth_res=63):
    depth_max = np.max(depth_map)
    depth_min = np.min(depth_map)
    depth_step = (depth_max - depth_min) / depth_res
    refocus_shifts = np.arange(depth_min, (depth_max+depth_step), depth_step)
    print(f"Min Depth: {depth_min}, Max Depth: {depth_max}, step: {depth_step}")
    print(refocus_shifts[0], refocus_shifts[-1])
    refocus_shifts = 1.0 - 1.0 / refocus_shifts
    print(refocus_shifts[0], refocus_shifts[-1])
    return refocus_shifts

def calculate_refocus_grad(light_field, interp_fns, refocus_shifts):
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