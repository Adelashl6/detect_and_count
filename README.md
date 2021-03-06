# detect_and_count
The codes are about keypoints-based anchor-free detection framework. Beside the three branches for predicting the center, scale, and
offset properties of the pedestrian in the image, we introduce another branch for regressing the pedestrian density and carefully design a regression loss guiding the training of model more robust to the crowd occlusion. Also we introduce a metric based on head counting to measure the robustness of our model to occlusion.

## Requirement
* Python 3.6
* Pytorch 0.4.1.post2
* Numpy 1.16.0
* OpenCV 3.4.2
* Torchvision 0.2.0

## Reproduction Environment
* Train new models: Two GPUs with 16G memory per GPU.(If you do not have enough GPU memory resources, please resize the input to 640x1280, it yields slightly worse performance, though.)



## Dataset
CityPersons Dataset

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
## Performance

R: 9.90%, HO:41.94%, R+HO: 24.64%
MAE: [Total: 12.10], [Num <= 10: 15.03], [10 < Num <= 30: 8.40], [Num >= 30: 5.04]



They should be stored in `./models/`.

## Compile Libraries

Before running the codes, you must compile the libraries. The followings should be accomplished in terminal. If you are not sure about what it means, click [here](https://linuxize.com/post/linux-cd-command/) may be helpful.

```
cd opr
make all
```


## Training

`python train.py`

or

`CUDA_VISIBLE_DEVICES=x,x python train.py --gpu_id 0 1`

## Test

`python test.py`

