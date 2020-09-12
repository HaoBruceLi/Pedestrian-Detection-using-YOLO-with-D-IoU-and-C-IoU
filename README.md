# Pedestrian-Detection-using-YOLO-with-D-IoU-and-C-IoU
This paper proposes a method to improve the performance of pedestrian detection. The method is based on the You Only Look Once (YOLO) algorithm and the improved Intersection over Union (IoU) loss function.
 
 
[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](LICENSE)

####  ENVIROMENT
- Python 3.5.2 or more 
- Keras 2.1.5
- tensorflow 1.6.0 or more

####  Structure

model_data------|---voc_annotation.py               # run before training voc dataset
                |---coco_annotation.py              # run before training coco dataset
                |---yolo_weights.h5                 # pre-trained weight file
                |---yolo_hao.h5                     # trained YOLO model      
                |---xxx_anchor and classess.txt     # traine different dataset  

Test------------|---result_imges                    # store tested images
                |---test_imges                      # images for testing
                |---yolo_test.py                    # test basic model performance for DEMO

yolo3-----------|---model.py                        # modify the training strategy
                |---utils.py                        # bottleneck training added in addtional 
                 
--------------------convert.py                      # Convert the Darknet YOLO model to a Keras model
--------------------darknet53.cfg                   # orignal network from Joseph
--------------------kmeans.py                       # k-means clustering and regression
--------------------train.py                        # train specific model
--------------------yolo.py                         # detecting configuration in images and videos
--------------------yolov3.cfg                      # Converted model from Darknet
--------------------yolo_video.py                   # more command line option parsing 




####  Quick Test

1. put test images to /Test/test_imges
2. run yolo_test.py
3. result stored in /Test/result_imges
       |||||
       vvvvv
==============================RUN IN TERMINAL==============================
cd ./Test
python3 yolo_test.py
===========================================================================


####  Introduction

A Keras implementation of YOLOv3 with modifiled loss function(Tensorflow backend).


####  Quick Start

1. Download YOLOv3 weights from [YOLO website](https://pjreddie.com/darknet/yolo/).
2. Convert the Darknet YOLO model to a Keras model. 
3. Run YOLO detection.
       |||||
       vvvvv
==============================RUN IN TERMINAL==============================
wget https://pjreddie.com/media/files/yolov3.weights
python3 convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
python3 yolo_video.py [OPTIONS...] --image, for image detection mode, OR
python3 yolo_video.py [video_path] [output_path (optional)]
===========================================================================

For another version of YOLO, just do in a similar way, just specify model path and anchor path with `--model model_file` and `--anchors anchor_file`.


### Usage
Use --help to see usage of yolo_video.py:
```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional arguments
```
---

MultiGPU usage: use `--gpu_num N` to use N GPUs. It is passed to the [Keras multi_gpu_model()](https://keras.io/utils/#multi_gpu_model).

####  Training

1. Generate your own annotation file and class names file.  
    One row for one image;  
    Row format: `image_file_path box1 box2 ... boxN`;  
    Box format: `x_min,y_min,x_max,y_max,class_id` (no space).  
    For VOC dataset, try `python voc_annotation.py`  
    Here is an example:
    ```
    path/to/img1.jpg 50,100,150,200,0 30,50,200,120,3
    path/to/img2.jpg 120,300,250,600,2
    ...
    ```

2. Make sure you have run `python convert.py -w yolov3.cfg yolov3.weights model_data/yolo_weights.h5`  
    The file model_data/yolo_weights.h5 is used to load pretrained weights.

3. Modify train.py and start training.  
    `python3 train.py`  
    Use your trained weights or checkpoint weights with command line option `--model model_file` when using yolo_video.py
    Remember to modify class path or anchor path, with `--classes class_file` and `--anchors anchor_file`.

If you want to use original pretrained weights for YOLOv3:  
    1. `wget https://pjreddie.com/media/files/darknet53.conv.74`  
    2. rename it as darknet53.weights  
    3. `python convert.py -w darknet53.cfg darknet53.weights model_data/darknet53_weights.h5`  
    4. use model_data/darknet53_weights.h5 in train.py

---



Thanks the author for sharing the Python YOLOv3 source code, the url: https://github.com/qqwweee/keras-yolo3
