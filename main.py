import argparse
import os
import numpy as np
import time
import scipy.io as sio
import shutil
from tqdm import tqdm
from PIL import Image
from embedding import embedding
from extraction import extraction
from skimage.measure import compare_psnr as psnr
from utils import my_nc

# ---------------------------- Parser arguments -----------------
ap = argparse.ArgumentParser()
ap.add_argument('-data_path', type=str, help='Where to get the images for embedding/extraction')
ap.add_argument('-img_size', '--image_size', type=int, nargs='+', help='Which size of image, [height, width, channel]')
ap.add_argument('-process_name', choices=['embedding', 'extraction'], type=str, help='Embedding or extraction process')
ap.add_argument('-coef', '--coefficient', type=int, nargs='+', help='Which used DCT coefficients in our algorithm')
ap.add_argument('-thresh', '--threshold', default=0.01, type=float, help='Which used threshold in our algorithm')
ap.add_argument('-blk_size', '--block_size', type=int, help='Which block size of image')
ap.add_argument('-cap', '--capacity', default=0.012, type=float, help='Capacity of watermarking as bit per pixel')
ap.add_argument('-seg_path', '--segmentation_model_path', type=str, help='Where to get the segmentation model')
ap.add_argument('-class_path', '--classification_model_path', type=str, help='Where to get the distortion detection model')
args = ap.parse_args()

data_path = args.data_path
img_size = args.image_size
process_name = args.process_name
coefficient = args.coefficient
thresh = args.threshold
block_size = args.block_size
capacity = args.capacity
segment_model_path = args.segmentation_model_path
class_model_path = args.classification_model_path
# --------------------------- Embedding process --------
if process_name == 'embedding':
    files = os.listdir(data_path)
    shutil.rmtree('workspace/img_marked', ignore_errors=True)
    os.makedirs('workspace/img_marked')  # Path of watermarked image

    # --------- Generate the random binary watermark -----------
    total_bit = int(capacity*np.prod(img_size))
    mark = np.random.randint(2, size=(total_bit, 1), dtype='uint8')

    switched_block = []  # Percent of switched NROI block into ROI block
    start_time = time.time()
    for file in tqdm(files):
        img_org = Image.open(os.path.join(data_path, file))
        img_org = np.asarray(img_org)
        [img_marked,  switched] = embedding(img_org, img_size, mark, block_size, thresh, coefficient,
                                            segment_model_path)  # embedding
        switched_block.append(switched)
        img_marked = Image.fromarray(img_marked)
        img_marked.save(os.path.join('workspace/img_marked', file))
    elapsed = time.time() - start_time
    sio.savemat('workspace/mark_'+str(capacity)+'.mat', {'mark': mark})
    print('Average percent of switched NROI block into ROI block : ' + str(np.mean(switched_block)))
    print('Embedding time: ', elapsed)

# ------------------------------------ Extraction process-----------
elif process_name == 'extraction':
    files = os.listdir(data_path)
    shutil.rmtree('workspace/img_recovered', ignore_errors=True)
    os.makedirs('workspace/img_recovered')
    mark_orig = sio.loadmat('./workspace/mark_' + str(capacity) + '.mat')['mark']

    start_time = time.time()
    for file in tqdm(files):
        img_marked = Image.open(os.path.join(data_path, file))
        img_marked = np.asarray(img_marked)
        [img_recovered, mark_extr] = extraction(img_marked, img_size, block_size, thresh, coefficient,
                                                capacity, segment_model_path, class_model_path)  # Extraction
        img_recovered = Image.fromarray(img_recovered)
        img_recovered.save(os.path.join('workspace/img_recovered', file))
        assert (my_nc(mark_extr, mark_orig) == 1)
    elapsed = time.time() - start_time
    print('Extraction time: ', elapsed)

    # ----------------- Evaluate ---------
    PSNR_NROI_marked, PSNR_NROI_recovered, PSNR_ROI_marked, PSNR_ROI_recovered, PSNR_img_marked, PSNR_img_recovered = \
        [[] for _ in range(6)]
    NROI_img_orig, NROI_img_marked, NROI_img_recovered, ROI_img_orig, ROI_img_marked, ROI_img_recovered = \
        [[] for _ in range(6)]
    gtruth_extension = os.path.splitext(os.listdir('workspace/img_gtruth')[0])[1]  # Extension of gtruth image
    for file in tqdm(files):
        img_marked = Image.open(os.path.join(data_path, file))
        img_marked = np.asarray(img_marked)
        img_orig = Image.open(os.path.join('workspace/img_orig', file))
        img_orig = np.asarray(img_orig)
        img_gtruth = Image.open(os.path.join('workspace/img_gtruth', os.path.splitext(file)[0] + gtruth_extension))
        img_gtruth = np.asarray(img_gtruth)
        img_recovered = Image.open(os.path.join('workspace/img_recovered', file))
        img_recovered = np.asarray(img_recovered)
        img_height, img_width = img_gtruth.shape
        for m in range(0, img_height):
            for n in range(0, img_width):
                if img_gtruth[m, n] == 0:  # NROI pixel
                    NROI_img_orig.append(img_orig[m, n])
                    NROI_img_marked.append(img_marked[m, n])
                    NROI_img_recovered.append(img_recovered[m, n])
                else:  # ROI pixel
                    ROI_img_orig.append(img_orig[m, n])
                    ROI_img_marked.append(img_marked[m, n])
                    ROI_img_recovered.append(img_recovered[m, n])
        PSNR_NROI_marked.append(psnr(np.array(NROI_img_orig), np.array(NROI_img_marked), data_range=255))
        PSNR_NROI_recovered.append(psnr(np.array(NROI_img_orig), np.array(NROI_img_recovered), data_range=255))
        PSNR_ROI_marked.append(psnr(np.array(ROI_img_orig), np.array(ROI_img_marked), data_range=255))
        PSNR_ROI_recovered.append(psnr(np.array(ROI_img_orig), np.array(ROI_img_recovered), data_range=255))
        PSNR_img_marked.append(psnr(img_orig, img_marked, data_range=255))
        PSNR_img_recovered.append(psnr(img_orig, img_recovered, data_range=255))
    print('Mean PSNR of watermarked images: ' + str(np.mean(PSNR_img_marked)))
    print('Mean PSNR of recovered images: ' + str(np.mean(PSNR_img_recovered)))
    print('Mean PSNR improvement: ' + str(np.abs(np.mean(PSNR_img_marked) - np.mean(PSNR_img_recovered))))
    print('Mean PSNR of NROI blocks before recovery: ' + str(np.mean(PSNR_NROI_marked)))
    print('Mean PSNR of NROI blocks after recovery: ' + str(np.mean(PSNR_NROI_recovered)))
    print('Mean NROI PSNR improvement: ' + str(np.abs(np.mean(PSNR_NROI_marked) - np.mean(PSNR_NROI_recovered))))
    print('Mean PSNR of ROI block before recovery: ' + str(np.mean(PSNR_ROI_marked)))
    print('Mean PSNR of ROI block after recovery: ' + str(np.mean(PSNR_ROI_recovered)))
    print('Mean ROI PSNR improvement: ' + str(np.abs(np.mean(PSNR_ROI_marked) - np.mean(PSNR_ROI_recovered))))
