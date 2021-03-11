import cv2
import os
import tensorflow as tf
from utils import *
from copy import deepcopy

model_from_json = tf.keras.models.model_from_json


def extraction(img_marked, img_size, block_size, th, coef, cap, segment_model_pth, class_model_path):
    if img_marked.ndim == 2:  # Grayscale image
        img_marked = np.expand_dims(img_marked, -1)
        img_marked = np.expand_dims(img_marked, 0)
    elif img_marked.ndim == 3:  # RGB image
        img_marked = np.expand_dims(img_marked, 0)
    img_height, img_width, img_channel = img_size

    # -------------------------------- Load the segmentation model and its weights
    with open(os.path.join(segment_model_pth, 'architecture_segmentation.json'), 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    json_file.close()
    model.load_weights(os.path.join(segment_model_pth, 'best_weights_segmentation.h5'))

    # -------------------- Divide original image into blocks -------
    blocks_img_marked = extract_blocks(img=img_marked, block_height=block_size, block_width=block_size)

    # -------------------------------- Segment blocks ---------------
    predictions = model.predict(blocks_img_marked)
    # print("predicted image size: ", predictions.shape)

    # ------------------------------- Convert the prediction arrays in corresponding blocks -------
    predict_blocks = pred_to_imgs(predictions, block_size, block_size, "threshold")

    # -------------------------------- Construct the segmented image with the segmented blocks ------
    predict_marked = recompone(predict_blocks, int(np.ceil((img_height/block_size))),
                               int(np.ceil((img_width/block_size))))
    predict_marked = predict_marked[:, 0:img_height, 0:img_width, :]

    # -------------------------- Extraction and recovery process --------------
    [u, v] = coef  # DCT coefficients
    total_bit = int(cap * np.prod(img_size))  # Size of watermark

    # -------------------------------- Load the classification model and its weights ------
    with open(os.path.join(class_model_path, 'architecture_classification.json'), 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    json_file.close()
    model.load_weights(os.path.join(class_model_path, 'best_weights_classification.h5'))

    mark = np.zeros((total_bit, 1), dtype='uint8')  # Extracted watermark
    img_recovered = deepcopy(img_marked)  # Recovered image

    cnt_mark = 0  # Counter of extracted watermark
    for m in range(0, block_size*(img_height//block_size), block_size):
        if cnt_mark == total_bit:  # Whole watermark has extracted
            break
        for n in range(0, block_size*(img_width//block_size), block_size):
            if cnt_mark == total_bit:
                break
            if np.sum(predict_marked[0, m:m+block_size, n:n+block_size, 0]) == 0:  # NROI block
                for chn in range(0, img_channel):
                    if cnt_mark == total_bit:
                        break
                    cover = img_marked[0, m:m+block_size, n:n+block_size, chn] / 255.
                    cover_dct = cv2.dct(cover)
                    prediction = model.predict(img_marked[0:1, m:m+block_size, n:n+block_size, chn:chn+1] / 255.)
                    if cover_dct[u, v] >= cover_dct[v, u]:
                        mark[cnt_mark] = 0
                        if np.round(prediction) == 1:  # Classifier detected that block has distorted during embedding
                            cover_dct[u, v] -= th
                            if cover_dct[u, v] > cover_dct[v, u]:
                                cover_dct[u, v], cover_dct[v, u] = cover_dct[v, u], cover_dct[u, v]
                            rec = cv2.idct(cover_dct) * 255.
                            for p in range(0, block_size):
                                for q in range(0, block_size):
                                    if rec[p, q] > 255:
                                        rec[p, q] = 255
                                    elif rec[p, q] < 0:
                                        rec[p, q] = 0
                            img_recovered[0, m:m+block_size, n:n+block_size, chn] = np.round(rec)
                    else:
                        mark[cnt_mark] = 1
                        if np.round(prediction) == 1:  # Classifier detected that block has distorted during embedding
                            cover_dct[v, u] -= th
                            if cover_dct[v, u] > cover_dct[u, v]:
                                cover_dct[u, v], cover_dct[v, u] = cover_dct[v, u], cover_dct[u, v]
                            rec = cv2.idct(cover_dct) * 255.
                            for p in range(0, block_size):
                                for q in range(0, block_size):
                                    if rec[p, q] > 255:
                                        rec[p, q] = 255
                                    elif rec[p, q] < 0:
                                        rec[p, q] = 0
                            img_recovered[0, m:m+block_size, n:n+block_size, chn] = np.round(rec)
                    cnt_mark += 1
    return np.squeeze(img_recovered), mark
