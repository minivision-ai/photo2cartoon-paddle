<div align='center'>
  <img src='./images/title.png'>
</div>

# Photo to Cartoon

[中文版](README.md) | English Version

[Minivision](https://www.minivision.cn/)'s photo-to-cartoon translation project is opened source in this repo, you can try our WeChat mini program "AI Cartoon Show" via scanning the QR code below.

<div>
  <img src='./images/QRcode.jpg' height='150px' width='150px'>
</div>

You can also try on this page: [https://ai.minivision.cn/#/coreability/cartoon](https://ai.minivision.cn/#/coreability/cartoon)

## Introduce

The aim of portrait cartoon stylization is to transform real photos into cartoon images with portrait's ID information and texture details. We use Generative Adversarial Network method to realize the mapping of picture to cartoon. Considering the difficulty in obtaining paired data and the non-corresponding shape of input and output, we adopt unpaired image translation fashion.

The results of CycleGAN, a classic unpaired image translation method, often have obvious artifacts and are unstable. Recently, Kim et al. propose a novel normalization function (AdaLIN) and an attention module in paper "U-GAT-IT" and achieve exquisite selfie2anime results.

Different from the exaggerated anime style, our cartoon style is more realistic and contains unequivocal ID information. To this end, we add a Face ID Loss (cosine distance of ID features between input image and cartoon image) to reach identity invariance. 

We propose a Soft Adaptive Layer-Instance Normalization (Soft-AdaLIN) method which fuses the statistics of encoding features and decoding features in de-standardization. 

Based on U-GAT-IT, two hourglass modules are introduced before encoder and after decoder to improve the performance in a progressively way.

We also pre-process the data to a fixed pattern to help reduce the difficulty of optimization. For details, see below.

<div align='center'>
  <img src='./images/results.png'>
</div>

## Start

### Requirements
- python 3.6
- paddlepaddle-gpu 2.0.0rc0
- face-alignment
- dlib

### Clone

```
git clone https://github.com/minivision-ai/photo2cartoon-paddle.git
cd ./photo2cartoon
```

### Download

[Google Drive](https://drive.google.com/file/d/12M221441UU_7OB1u3JYAh9ENUa0v6BQZ/view?usp=sharing) | [Baidu Cloud](https://pan.baidu.com/s/17xt0gwHGd3tD2-aRzW4-8w) acess code: 9dnx

1. Put the pre-trained photo2cartoon model **photo2cartoon_weights.pdparams** into `models` folder.
2. Place the head segmentation model **seg_model_384.pdparams** in `utils/segment` folder.
3. Open-source cartoon dataset **`cartoon_data/`** contains `trainB` and `testB`.

### Test

Please use a young Asian woman photo.
```
python test.py --photo_path ./images/photo_test.jpg --save_path ./images/cartoon_result.png
```

### Train
**1.Data**

Training data contains portrait photos (domain A) and cartoon images (domain B). The following process can help reduce the difficulty of optimization.
- Detect face and its landmarks.
- Face alignment according to landmarks.
- expand the bbox of landmarks and crop face.
- remove the background by semantic segment.

<div align='center'>
  <img src='./images/data_process.jpg'>
</div>

We provide 204 cartoon images, besides, you need to prepare about 1,000 young Asian women photos and pre-process them by following command.

```
python data_process.py --data_path YourPhotoFolderPath --save_path YourSaveFolderPath
```

The `dataset` directory should look like this:
```
├── dataset
    └── photo2cartoon
        ├── trainA
            ├── xxx.jpg
            ├── yyy.png
            └── ...
        ├── trainB
            ├── zzz.jpg
            ├── www.png
            └── ...
        ├── testA
            ├── aaa.jpg 
            ├── bbb.png
            └── ...
        └── testB
            ├── ccc.jpg 
            ├── ddd.png
            └── ...
```

**2.Train**

Train from scratch:
```
python train.py --dataset photo2cartoon
```

Load pre-trained weights:
```
python train.py --dataset photo2cartoon --pretrained_weights models/photo2cartoon_weights.pdparams
```

## Q&A
#### Q：Why is the result of this project different from mini program?

A: For better performance, we customized the cartoon data (about 200 images) when training model for mini program. We also improved input size for high definition. Besides, we adopted our internal recognition model to calculate Face ID Loss.

#### Q: How to select best model?

A: We trained model about 300k iterations, then selected best model according to FID metric.

#### Q：Can I use the segmentation model to predict half-length portrait?
A：No. The model is trained for croped face specifically.

## Reference

U-GAT-IT: Unsupervised Generative Attentional Networks with Adaptive Layer-Instance Normalization for Image-to-Image Translation [[Paper](https://arxiv.org/abs/1907.10830)][[Code](https://github.com/znxlwm/UGATIT-pytorch)]

PaddleSeg: End-to-End Image Segmentation Kits Based on PaddlePaddle [Progect](https://github.com/PaddlePaddle/PaddleSeg)

PaddleGAN: PaddlePaddle GAN library and applications [Progect](https://github.com/PaddlePaddle/PaddleGAN)

