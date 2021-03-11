import cv2
import os
import tensorflow as tf
from utils import *
from copy import deepcopy

model_from_json = tf.keras.models.model_from_json


def embedding(img_orig, img_size, mark, block_size, th, coef, segment_model_path):
    if img_orig.ndim == 2:  # Grayscale image
        img_orig = np.expand_dims(img_orig, -1)
        img_orig = np.expand_dims(img_orig, 0)
    elif img_orig.ndim == 3:  # RGB image
        img_orig = np.expand_dims(img_orig, 0)
    img_height, img_width, img_channel = img_size

    # -------------------------------- Load the segmentation model and its weights
    with open(os.path.join(segment_model_path, 'architecture_segmentation.json'), 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    json_file.close()
    model.load_weights(os.path.join(segment_model_path, 'best_weights_segmentation.h5'))

    # -------------------- Divide original image into blocks -------
    blocks_img_orig = extract_blocks(img=img_orig, block_height=block_size, block_width=block_size)

    # -------------------------------- Segment blocks ---------------
    predictions = model.predict(blocks_img_orig)
    # print("predicted images size: ", predictions.shape)

    # -------------------------------- Convert the prediction arrays in corresponding blocks -------
    pred_blocks = pred_to_imgs(predictions, block_size, block_size, "threshold")

    # -------------------------------- Construct the segmented image with the segmented blocks ------
    pred_orig = recompone(pred_blocks, int(np.ceil((img_height/block_size))),
                          int(np.ceil((img_width/block_size))))      # predictions
    pred_orig = pred_orig[:, 0:img_height, 0:img_width, :]

    # ------------------------------- Embedding process -----------------------
    [u, v] = coef  # DCT coefficients
    total_bit = mark.size  # Size of watermark

    img_marked = deepcopy(img_orig)  # Watermarked image
    pred = deepcopy(pred_orig[0, :, :, 0])  # Segmented image before embedding
    total_switch = 0  # Total number of switched NROI block into ROI block

    while True:  # Embed until ROI block map remains unchanged
        cnt_mark = 0  # Counter of embedded watermark
        for m in range(0, block_size*(img_height//block_size), block_size):
            if cnt_mark == total_bit:  # Whole watermark has embedded
                break
            for n in range(0, block_size*(img_width//block_size), block_size):
                if cnt_mark == total_bit:  # whole watermark has embedded
                    break
                if np.sum(pred[m:m+block_size, n:n+block_size]) == 0:  # NROI block
                    for chn in range(0, img_channel):  # Embedding throughout channels
                        if cnt_mark == total_bit:  # Whole watermark has embedded
                            break
                        cover = img_orig[0, m:m+block_size, n:n+block_size, chn] / 255.
                        cover_dct = cv2.dct(cover)  # Apply DCT
                        if (mark[cnt_mark] == 0) and (cover_dct[u, v] <= cover_dct[v, u]):
                            cover_dct[u, v], cover_dct[v, u] = cover_dct[v, u], cover_dct[u, v]
                            cover_dct[u, v] += th
                        elif (mark[cnt_mark] == 1) and (cover_dct[v, u] <= cover_dct[u, v]):
                            cover_dct[u, v], cover_dct[v, u] = cover_dct[v, u], cover_dct[u, v]
                            cover_dct[v, u] += th
                        marked = cv2.idct(cover_dct)  # Apply inverse DCT
                        cnt_mark += 1
                        marked = marked * 255.
                        marked = np.round(marked)
                        for p in range(0, block_size):
                            for q in range(0, block_size):
                                if marked[p, q] > 255:  # Overflow
                                    marked[p, q] = 255
                                elif marked[p, q] < 0:  # Underflow
                                    marked[p, q] = 0
                        img_marked[0, m:m+block_size, n:n+block_size, chn] = marked

        # -------------------------- Divide watermarked image into blocks -------
        blocks_img_marked = extract_blocks(img=img_marked, block_height=block_size, block_width=block_size)

        # -------------------------------- Segment blocks ---------------
        predictions = model.predict(blocks_img_marked)

        # -------------------------------- Convert the prediction arrays in corresponding blocks -------
        pred_blocks = pred_to_imgs(predictions, block_size, block_size, "threshold")

        # -------------------------------- Construct the segmented image with the segmented blocks ------
        pred_marked = recompone(pred_blocks, int(np.ceil((img_height/block_size))),
                                int(np.ceil((img_width/block_size))))
        pred_marked = pred_marked[:, 0:img_height, 0:img_width, :]

        cnt_switch = 0  # Number of switched NROI block into ROI block in this iteration
        for m in range(0, block_size*(img_height//block_size), block_size):
            for n in range(0, block_size*(img_width//block_size), block_size):
                if np.sum(pred[m:m+block_size, n:n+block_size]) == 0:
                    if np.sum(pred_marked[0, m:m+block_size, n:n+block_size, 0]) != 0:
                        cnt_switch += 1
        pred = np.logical_or(pred_marked[0, :, :, 0], pred)
        assert(total_bit == cnt_mark)
        if cnt_switch == 0:  # ROI block map remain unchanged
            break
        else:
            total_switch += cnt_switch

    # ------------------------------- Calculate percent of switched NROI block into ROI block --------
    cnt_nroi = 0  # Total number of NROI blocks
    for m in range(0, block_size * (img_height // block_size), block_size):
        for n in range(0, block_size * (img_width // block_size), block_size):
            if np.sum(pred_orig[0, m:m + block_size, n:n + block_size, 0]) == 0:
                cnt_nroi += 1
    switched_blk = (total_switch * 100) / cnt_nroi

    return img_marked.squeeze(), switched_blk
