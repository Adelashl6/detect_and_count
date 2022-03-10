# detect_and_count
The codes are about keypoints-based anchor-free detection framework. Beside the three branches for predicting the center, scale, and36
offset properties of the pedestrian in the image, we introduce another branch for regressing the pedestrian density and carefully design a regression loss guiding the training of model more robust to the crowd occlusion. Also we introduce a metric based on head counting to measure the robustness of our model to occulusion.

## Requirement
* Python 3.6
* Pytorch 0.4.1.post2
* Numpy 1.16.0
* OpenCV 3.4.2
* Torchvision 0.2.0

## Reproduction Environment
* Train new models: Two GPUs with 16G memory per GPU.(If you do not have enough GPU memory resources, please resize the input to 640x1280, it yields slightly worse performance, though.)



## Preparation
1. CityPersons Dataset

You should download the dataset from [here](https://www.cityscapes-dataset.com/downloads/). From that link, leftImg8bit_trainvaltest.zip (11GB) is used. We use the training set(2975 images) for training and the validation set(500 images) for test. The data should be stored in `./data/citypersons/images`. Annotations have already prepared for you. And the directory structure will be 
```
*DATA_PATH
	*images
		*train
			*aachen
			*bochum
			...
		*val
			*frankfurt
			*lindau
			*munster
		*test
			*berlin
			*bielefeld
			...
	*annotations
		*anno_train.mat
		*anno_val.mat
		...
```
2. Wideface Dataset
3. Crowd HUman Dataset


2. Pretrained Models

The backbone of our ACSP is modified ResNet-101, i.e. replacing all BN layers with SN layers. You can download the pretrained model from [here](https://pan.baidu.com/s/1rK-ukAjEIPql2ECi38hRbQ). It is provided by the author of [Switchable Normalization](https://github.com/switchablenorms/Switchable-Normalization). The weights will be stored in `./models/`.

3. Our Trained Models

We provide two models:

[ACSP(Smooth L1)](https://pan.baidu.com/s/1p2IF7nI6dOhpmSvXLFsxlA)(code: ydc1): **Reasonable 10.0%; Heavy 46.1%; Partial 8.8%; Bare 6.7%**.

[ACSP(Vanilla L1)](https://pan.baidu.com/s/1zZP3brc1FvMrcmPo7Fx-Tg)(code: 4oa2): **Reasonable 9.3%; Heavy 46.3%; Partial 8.7%; Bare 5.6%**.

They should be stored in `./models/`.

4. Compile Libraries

Before running the codes, you must compile the libraries. The followings should be accomplished in terminal. If you are not sure about what it means, click [here](https://linuxize.com/post/linux-cd-command/) may be helpful.

```
cd util
make all
```


## Training

`python train.py`

or

`CUDA_VISIBLE_DEVICES=x,x python train.py --gpu_id 0 1`

## Test

`python test.py`

