## BlessMark: A Blind Diagnostically-Lossless Watermarking Framework
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)
[![TensorFlow 1.13.1](https://img.shields.io/badge/TensorFlow-1.13.1-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v1.13.1)


![Im1](https://drive.google.com/uc?export=view&id=17W-PxuEeyBmO9G1MX9yOnMqEETNGJSOb)
Block diagram of BlessMark framework, composed of three separate modules for embedding of the watermark, extraction of the watermark, and recovery of the original cover image.

This repository contains the source code for the method described in the paper:
> **BlessMark: A Blind Diagnostically-Lossless Watermarking Framework for Medical Applications Based on Deep Neural Networks**
> 
> *Multimedia Tools and Applications, Springer, 2020*  
> Hamidreza Zarrabi, Ali Emami, Pejman Khadivi, Nader Karimi, Shadrokh Samavi</br>
> **[[Paper]](https://link.springer.com/article/10.1007%2Fs11042-020-08698-9)**
> 
> **Abstract:** *Nowadays, with the development of public network usage, medical information is transmitted throughout the hospitals. A watermarking system can help for the confidentiality of medical information distributed over the internet. In medical images, regions-of-interest (ROI) contain diagnostic information. The watermark should be embedded only into non-regions-of-interest (NROI) regions to keep diagnostically important details without distortion. Recently, ROI based watermarking has attracted the attention of the medical research community. The ROI map can be used as an embedding key for improving confidentiality protection purposes. However, in most existing works, the ROI map that is used for the embedding process must be sent as side-information along with the watermarked image. This side information is a disadvantage and makes the extraction process non-blind. Also, most existing algorithms do not recover NROI of the original cover image after the extraction of the watermark. In this paper, we propose a framework for blind diagnostically-lossless watermarking, which iteratively embeds only into NROI. The significance of the proposed framework is in satisfying the confidentiality of the patient information through a blind watermarking system, while it preserves diagnostic/medical information of the image throughout the watermarking process. A deep neural network is used to recognize the ROI map in the embedding, extraction, and recovery processes. In the extraction process, the same ROI map of the embedding process is recognized without requiring any additional information. Hence, the watermark is blindly extracted from the NROI. Furthermore, a three-layer fully connected neural network is used for the detection of distorted NROI blocks in the recovery process to recover the distorted NROI blocks to their original form. The proposed framework is compared with one lossless watermarking algorithm. Experimental results demonstrate the superiority of the proposed framework in terms of side information.*

## Motivation
Why this method exist while there are many watermarking methods? Since BlessMark method is:
- A framework that can be used as a platform for applying a desired watermarking method.
- A non-blind method which same ROI map is recognized in embedding and extraction processes.
- A diagnostically lossless method which the NROI blocks recovered to their original state wherever possible.

## Installation
This code was tested with tensorflow 1.13.1, python 3.6 and ubuntu 20.04.
### 1. Install Anaconda
To install Anaconda [just follow the tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
### 2. Install Requirements
Just run it in the terminal
```bash
conda create -n BlessMark python=3.6
conda activate BlessMark
conda install -c anaconda tensorflow-gpu=1.13.1
git clone https://github.com/HamidrezaZarrabi/BlessMark.git
cd BlessMark
python -m pip install -r requirements.txt
```
### 3. Download Pretrain (optional)
To fair comparison, we recommend utilizing the pre-trained models if they are used for academic purpose. The link to the pretrained segmentation and classification models of [Angiography dataset](https://drive.google.com/file/d/1GdpKQuV39xewTdX3Xu5LcXsw0ao9F43f/view?usp=sharing)
<table align="center">
<tr>
<td align="center">Block Size</td><td align="center">6x6</td><td align="center">8x8</td><td align="center">10x10</td>
</tr>
<tr>
<td align="center">Network</td>
<td align="center"><a href="https://drive.google.com/file/d/1R91QlcuD0JrFNlFbquNcUHCquJl-gBlh/view?usp=sharing"><div>Segmentation</div></a>
<a href="https://drive.google.com/file/d/11dP4b0jPEh5XnjUCFwb_8i7UP0V2vW4F/view?usp=sharing"><div>Classification</div></a></td>
<td align="center"><a href="https://drive.google.com/file/d/1x9KDJ4GG3wTrfXmzLTuEoP-jI-bchqLr/view?usp=sharing"><div>Segmentation</div></a>
<a href="https://drive.google.com/file/d/1iILqK1SndeNwqKiMRU5DBAnPMvadOnnz/view?usp=sharing"><div>Classification</div></a></td>
<td align="center"><a href="https://drive.google.com/file/d/1vyw8BA81sHfy93Y6jQZoJ6VFrgftA0Ti/view?usp=sharing"><div>Segmentation</div></a>
<a href="https://drive.google.com/file/d/1yx0cx8pB71qVmQ1bg4YO1S_nVqyBnnBM/view?usp=sharing"><div>Classification</div></a></td>
</tr>
</table>

## Inference
Just run it in the terminal. To perform embedding or extraction, use
```bash
python main.py
```
There are several arguments that must be used, which are
```
-data_path +str #Where to get the images for embedding/extraction
-image_size +nargs #Which size of image as [height, width, channel]
-process_name +str choices=['embedding', 'extraction'] #Embedding or extraction process
-coefficient +nargs #Which used DCT coefficients in our algorithm
-thresh +float #Which used threshold in our algorithm
-block_size +int #Which block size of image
-capacity +float #Capacity of watermarking as bit per pixel
-segmentation_model_path +str #Where to get the segmentation model
-classification_model_path +str #Where to get the classification model
```
For example, to embed watermark
```bash
python main.py -data_path workspace/img_orig -img_size 512 512 1 -process_name embedding -coef 6 7 -thresh 0.01 -blk_size 8 -cap 0.012 -seg_path workspace/segmentation_8x8
```
the watermarked image is saved in `workspace/img_marked/` directory.<br/>
For example, to extract watermark and recovery original image
```bash
python main.py -data_path workspace/img_marked -img_size 512 512 1 -process_name extraction -coef 6 7 -thresh 0.01 -blk_size 8 -cap 0.012 -seg_path workspace/segmentation_8x8 -class_path workspace/classification_8x8
```
the recovered image is saved in `workspace/img_recovered/` directory.

## Results (From Pretrained models)
Some results on 22 images of [Angiography dataset](https://drive.google.com/file/d/1GvwetKvA0jOo6uu95MZ4sWZs6KuuG5fI/view?usp=sharing).
<table align="center" style="margin: 0px auto;">
<tr>
<td> Block size</td><td align="center">6x6</td><td align="center">8x8</td>
</tr>
<tr>
<td>Capacity (BPP)</td> <td>0.022 </td><td> 0.012 </td>
</tr>
<tr>
<td>PSNR of Watermarked Image (dB)</td> <td>55.65 </td><td>58.20 </td>
</tr>
<tr>
<td> PSNR of Recovered Image (dB) <td>62.81 </td><td> 65.32 </td>
</tr>
<tr>
<td>Improvement of PSNR(dB)</td><td>7.16 </td> <td>7.12 </td>
</tr>
<tr>
<td>PSNR of Watermarked Image NROI (dB)</td> <td>55.48 </td><td> 57.99 </td>
</tr>
<tr>
<td>PSNR of Recovered Image NROI (dB)</td> <td>62.66 </td><td>65.11 </td>
</tr>
<tr>
<td>Improvement of NROI PSNR (dB)</td><td>7.18 </td><td>7.12 </td>
</tr>
<tr>
<td>PSNR of Watermarked Image ROI (dB) </td> <td>59.63 </td><td>63.73</td>
</tr>
<tr>
<td>PSNR of Recovered Image ROI (dB)</td><td>65.96 </td><td> 70.88 </td>
</tr>
<tr>
<td>Improvement of ROI PSNR (dB)</td><td>6.33 </td><td>7.15 </td>
</tr>
</table>

## Citation
Please cite our paper if you find this repo useful in your research.
```
@article{Zarrabi_2020,
	doi = {10.1007/s11042-020-08698-9},
	url = {https://doi.org/10.1007%2Fs11042-020-08698-9},
	year = 2020,
	month = {may},
	publisher = {Springer Science and Business Media {LLC}},
	volume = {79},
	number = {31-32},
	pages = {22473--22495},
	author = {Hamidreza Zarrabi and Ali Emami and Pejman Khadivi and Nader Karimi and Shadrokh Samavi},
	title = {{BlessMark}: a blind diagnostically-lossless watermarking framework for medical applications based on deep neural networks},
	journal = {Multimedia Tools and Applications}
}
```
