# YOLOv4-CSP

This folder contains a modified version of YOLOv4-CSP, [Scaled-YOLOv4: Scaling Cross Stage Partial Network](https://arxiv.org/abs/2011.08036)".

The remaining of this README is the original README from the YOLOv4-CSP repository.

## Installation

```
# create the docker container, you can change the share memory size if you have more.
nvidia-docker run --name yolov4_csp -it -v your_coco_path/:/coco/ -v your_code_path/:/yolo --shm-size=64g nvcr.io/nvidia/pytorch:20.11-py3

# apt install required packages
apt update
apt install -y zip htop screen libgl1-mesa-glx

# pip install required packages
pip install seaborn thop

# install mish-cuda if you want to use mish activation
# https://github.com/thomasbrandon/mish-cuda
# https://github.com/JunnYu/mish-cuda
cd /
git clone https://github.com/JunnYu/mish-cuda
cd mish-cuda
python setup.py build install

# go to code folder
cd /yolo
```

## Testing

[`yolov4-csp.weights`](https://drive.google.com/file/d/1TdKvDQb2QpP4EhOIyks8kgT8dgI1iOWT/view?usp=sharing)

<details><summary> <b>old weights</b> </summary>

[`yolov4-csp.weights`](https://drive.google.com/file/d/1NQwz47cW0NUgy7L3_xOKaNEfLoQuq3EL/view?usp=sharing)

</details>

```
# download yolov4-csp.weights and put it in /yolo/weights/ folder.
python test.py --img 640 --conf 0.001 --iou 0.65 --batch 8 --device 0 --data coco.yaml --cfg models/yolov4-csp.cfg --weights weights/yolov4-csp.weights
```

You will get the results:
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.48656
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.67002
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.52739
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.33082
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.54036
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.62107
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.37197
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.61211
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.66544
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.49676
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.72018
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.80528
```
<details><summary> <b>old results</b> </summary>

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.47827
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.66448
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.51928
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.30647
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.53106
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.61056
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.36823
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.60434
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.65795
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.48486
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.70892
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.79914
```

</details>

## Training

```
# you can change batch size to fit your GPU RAM.
python train.py --device 0 --batch-size 16 --data coco.yaml --cfg yolov4-csp.cfg --weights '' --name yolov4-csp
```

For resume training:
```
# assume the checkpoint is stored in runs/train/yolov4-csp/weights/.
python train.py --device 0 --batch-size 16 --data coco.yaml --cfg yolov4-csp.cfg --weights 'runs/train/yolov4-csp/weights/last.pt' --name yolov4-csp --resume
```

If you want to use multiple GPUs for training
```
python -m torch.distributed.launch --nproc_per_node 4 train.py --device 0,1,2,3 --batch-size 64 --data coco.yaml --cfg yolov4-csp.cfg --weights '' --name yolov4-csp --sync-bn
```

## Citation

```
@InProceedings{Wang_2021_CVPR,
    author    = {Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
    title     = {{Scaled-YOLOv4}: Scaling Cross Stage Partial Network},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13029-13038}
}
```
