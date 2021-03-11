import h5py
import numpy as np
from copy import deepcopy


# -------- My normalized correlation function
def my_nc(watermark_extr, watermark_orig):
    assert(np.shape(watermark_extr) == np.shape(watermark_orig))
    extr = deepcopy(watermark_extr)
    orig = deepcopy(watermark_orig)
    height, width = np.shape(orig)
    for m in range(0, height):
        for n in range(0, width):
            if orig[m, n] == 0:
                orig[m, n] = -1
            if extr[m, n] == 0:
                extr[m, n] = -1
    tmp = np.sum(orig * extr)
    tmp = tmp / watermark_orig.size
    tmp = tmp + 1
    nc = tmp / 2
    return nc


# --------------- Extract blocks from image ---------
def extract_blocks(img, block_height, block_width):
    # ---------- Convert RGB image to grayscale image and scale it to range [0, 1]
    assert img.ndim == 4
    if img.shape[3] == 3:  # RGB image
        img = rgb2gray(img)  # black-white conversion
        img = img / 255.  # Scale to 0-1 range
    else:
        img = img / 255.  # scale to 0-1 range

    img = paint_border(img, block_height, block_width)  # Extend image
    blocks_img = extract_ordered(img, block_height, block_width)  # Extract blocks
    return blocks_img


# ------------- Divide the image into blocks -------
def extract_ordered(full_img, block_h, block_w):
    assert (len(full_img.shape) == 4)  # 4D arrays
    assert (full_img.shape[3] == 1 or full_img.shape[3] == 3)  # Check the channel is 1 or 3
    img_h = full_img.shape[1]  # Height of the full image
    img_w = full_img.shape[2]  # Width of the full image

    n_blocks_h = int(img_h/block_h)  # Round to lowest int
    if img_h % block_h != 0:
        print("warning: " + str(n_blocks_h) + " blocks in height, with about " + str(img_h % block_h) +
              " pixels left over")
    n_blocks_w = int(img_w/block_w)  # Round to lowest int
    if img_w % block_w != 0:
        print("warning: " + str(n_blocks_w) + " blocks in width, with about " + str(img_w % block_w) +
              " pixels left over")
    n_blocks_tot = (n_blocks_h*n_blocks_w)*full_img.shape[0]
    blocks = np.empty((n_blocks_tot, block_h, block_w, full_img.shape[3]))

    iter_tot = 0  # Total number of blocks (N_blocks)
    for h in range(n_blocks_h):
        for w in range(n_blocks_w):
            block = full_img[0, h*block_h:(h*block_h)+block_h, w*block_w:(w*block_w)+block_w, :]
            blocks[iter_tot] = block
            iter_tot += 1
    assert (iter_tot == n_blocks_tot)
    return blocks  # Array with the full_img divided in blocks


# -------- Construct the image with the blocks ----------
def recompone(data, n_h, n_w):
    assert (data.shape[3] == 1 or data.shape[3] == 3)  # Check the channel is 1 or 3
    assert(len(data.shape) == 4)
    n_block_per_img = n_w*n_h
    assert(data.shape[0] % n_block_per_img == 0)
    n_full_img = int(data.shape[0]/n_block_per_img)
    block_h = data.shape[1]
    block_w = data.shape[2]
    # Define and start full recompone
    full_recomp = np.empty((n_full_img, n_h*block_h, n_w*block_w, data.shape[3]))
    k = 0  # Iter full img
    s = 0  # Iter single block
    while s < data.shape[0]:
        # Recompone one:
        single_recon = np.empty((n_h*block_h, n_w*block_w, data.shape[3]))
        for h in range(n_h):
            for w in range(n_w):
                single_recon[h*block_h:(h*block_h)+block_h, w*block_w:(w*block_w)+block_w, :] = data[s]
                s += 1
        full_recomp[k] = single_recon
        k += 1
    assert(k == n_full_img)
    return full_recomp


# ----------------- Extend the image because block division is not exact
def paint_border(data, block_h, block_w):
    assert (data.shape[3] == 1 or data.shape[3] == 3)  # Check the channel is 1 or 3
    img_h = data.shape[1]
    img_w = data.shape[2]

    if (img_h % block_h) == 0:
        new_img_h = img_h
    else:
        new_img_h = int(np.ceil(img_h/block_h))*block_h
    if (img_w % block_w) == 0:
        new_img_w = img_w
    else:
        new_img_w = int(np.ceil(img_w/block_w))*block_w
    new_data = np.zeros((data.shape[0], new_img_h, new_img_w, data.shape[3]))
    new_data[:, 0:img_h, 0:img_w, :] = data
    return new_data


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# -------- Convert RGB image into grayscale
def rgb2gray(rgb):
    assert rgb.ndim == 4  # 4D arrays
    bn_img = rgb[:, :, :, 0]*0.299 + rgb[:, :, :, 1]*0.587 + rgb[:, :, :, 2]*0.114
    bn_img = np.reshape(bn_img, (rgb.shape[0], rgb.shape[1], rgb.shape[2], 1))
    return bn_img


# ------------------ Convert the prediction arrays in corresponding blocks
def pred_to_imgs(pred, block_height, block_width, mode="original"):
    assert (len(pred.shape) == 3)  # 3D array: (n_blocks, height*width, 2)
    assert (pred.shape[2] == 2)  # Check the classes are 2
    pred_image = np.empty((pred.shape[0], pred.shape[1]))  # (n_blocks,height*width)
    if mode == "original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_image[i, pix] = pred[i, pix, 1]
    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i, pix, 1] >= 0.5:
                    pred_image[i, pix] = 1
                else:
                    pred_image[i, pix] = 0
    else:
        print("mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_image = np.reshape(pred_image, (pred_image.shape[0], block_height, block_width, 1))
    return pred_image
