# Real-Time Style Transfer in [TensorFlow](https://github.com/tensorflow/tensorflow)
This repository is Tensorflow implementation of Johnson's [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155). 

<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42068027-830719f4-7b84-11e8-9e87-088f1e476aab.png" width=800>
</p>
  
<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42068549-97588fc0-7b87-11e8-8110-93796a42a293.png" width=700>
</p>  

It takes 385 ms on a GTX1080Ti to style the MIT Stata Center (1024x680).

## Requirements
- tensorflow 1.18.0
- python 3.5.3  
- numpy 1.14.2  
- scipy 0.19.0
- moviepy 0.2.3.2

## Video Stylization
Here we transformed every frame in a video using various stylle images, then combined the results. [Click to go to the full demo on YouTube!](https://www.youtube.com/watch?v=HpKMLA19zkg&feature=youtu.be)

<p align = 'center'>
  <a href = 'https://www.youtube.com/watch?v=HpKMLA19zkg&feature=youtu.be'>
    <img src = 'https://user-images.githubusercontent.com/37034031/42082312-9ffdbffc-7bc2-11e8-9dfe-e505e5b3c528.gif'>
  </a>
</p>

## Image Stylization
A photo of Chicago was applied for various style paintings. Click on the ./examples/style folder to see full applied style images. **For more results you can find [here](https://www.dropbox.com/sh/pp6hmjrq9odekeq/AABzRxR2SZoDbWikMuu2VrKda?dl=0)**.

<p align="center">
<img src="https://user-images.githubusercontent.com/37034031/42069725-38fab690-7b8e-11e8-8deb-a63fbc09e3f6.png">
</p>

## Implementation Details
Implementation uses TensorFlow to rain a real-time style transfer network. Same transformation network is used as described in Johnson, except that batch normalization is replaced with Ulyanov's instance normalization, zero padding is replaced by reflected padding to reduce boundary artifacts, and the scaling/offset of the output `tanh` layer is slightly different.  

We follow  [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer) to use a loss function close to the one described in Gatys, using VGG19 instead of VGG16 and typically using "shallower" layers than in Johson's implementation (e.g. `relu1_1` is used rather than `relu1_2`).

## Documentation
### Training Style Transfer Networks
Use `main.py` to train a new style transform network. Training takes 6-8 hours on a GTX 1080Ti. **Before you run this, you should run `setup.sh`**. Example usage:

```
python main.py --style_img path/to/style/img.jpg \
  --train_path path/to/trainng/data/fold \
  --test_path path/to/test/data/fold \
  --vgg_path path/to/vgg19/imagenet-vgg-verydeep-19.mat
```
- `--gpu_index`:        gpu index, default: `0`  
- `--checkpoint_dir`:   dir to save checkpoint in, default: `./checkpoints`  
- `--style_img`:        style image path, default: `./examples/style/la_muse.jpg`  
- `--train_path`:       path to trainng images folder, default: `../Data/coco/img/train2014`  
- `--test_path`:        test image path, default: `./examples/content`  
- `--test_dir`:         test oa,ge save dor. default: `./examples/temp`    
- `--epochs`:           number of epochs for training data, default: `2`    
- `--batch_size`:       batch size for single feed forward, default: `4`    
- `--vgg_path`:         path to VGG19 network, default: `../Models_zoo/imagenet-vgg-verydeep-19.mat`  
- `--content_weight`:   content weight, default: `7.5`  
- `--style_weight`:     style weight, default: `100.`  
- `--tv_weight`:        total variation regularization weight, default: `200.`  
- `--print_freq`:       print loss frequency, default: `100`  
- `--sample_freq`:      sample frequency, default: `2000`  

### Evaluating Style Transfer Networks
Use `evaluate.py` to evaluate a style transfer network. Evaluation takes 300 ms per frame on a GTX 1080Ti. Takes several seconds per frame on a CPU. **Models for evaluation are [located here](https://www.dropbox.com/sh/3t7a4x8szw0vfw1/AABJkx2aouEw1mBOR73WUqZ5a?dl=0)**. Example usage:
```
python evaluate.py --checkpoint_dir path/to/checkpoint /
  --in_path path/to/test/image/folder
```
- `--gpu_index`: gpu index, default: `0`  
- `--checkpoint_dir`: dir to read checkpoint in, default: `./checkpoints/la_muse`  
- `--in_path`: test image path, default: `./examples/test`  
- `--out_path`: destination dir of transformed files, default: `./examples/results` 

### Stylizing Video
Use `transform_video.py` to transfer style into a video. Requires `moviepy`. Example usage:
```
python transform_video.py --checkpoint_dir path/to/checkpoint /
  --in_path path/to/input/video.mp4 /
  --out_path path/to/write/predicted_video.mp4
```
- `--gpu_index`: gpu index, default: `0`  
- `--checkpoint_dir`: dir to read checkpoint in, default: `./checkpoints/la_muse`  
- `--in_path`: input video path, default: `None`
- `--out_path`: path to save processed video to, default: `None`

### Citation
```
  @misc{chengbinjin2018realtimestyletransfer,
    author = {Cheng-Bin Jin},
    title = {Real-Time Style Transfer},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/Real-time-style-transfer/}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project borrowed some code from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer) adnd Anish's [Neural Style](https://github.com/anishathalye/neural-style/)
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)
- The image of the MIT Stata Center at the very beginning of the README was taken by [Juan Paulo](https://juanpaulo.me/)

## License
Copyright (c) 2016 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.
