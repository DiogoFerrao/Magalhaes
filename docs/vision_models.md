# Vision Models

In this page we describe the models available for object detection and the commands necessary to train, test, and generate detections.

## Available Models

We have three types of models available, the pretrained which are the publicly available weights, the distilled which are the models pre-trained using knowledge distillation, and the finetuned which are the models trained using the Schréder dataset. From this group only the finetuned support the set of classes of interest for this project. Jointly with the  finetuned models, we also provide their mAP on the test set and the respective cross validation results on a two fold split, all using the same thresholds (conf_thres=0.05 and iou_thres=0.55).

> **Note** the weights and the name of the finetuned models correspond to the model that was trained using the full dataset. The cross validation results correspond to identical models trained using a two fold split of the dataset.

**Pretrained models**

| Model       | Weights                                          |
| ----------- | ------------------------------------------------ |
| YOLOv4-CSP  | `magalhaes/vision/pretrained/yolov4-csp.weights` |
| YOLOv7      | `magalhaes/vision/pretrained/yolov7.pt`          |
| YOLOv7-tiny | `magalhaes/vision/pretrained/yolov7-tiny.pt`     |

**Distilled models**

| Name              | Model       | Weights                                                            |
| ----------------- | ----------- | ------------------------------------------------------------------ |
| distil_1686477792 | YOLOv7-tiny | `magalhaes/vision/distillations/distil_1686477792/weights/best.pt` |

**Finetuned models**

| Name                   | Model                   | mAP (Cross Val) | mAP (test) | Weights                                                               | ONNX                                                        |
| ---------------------- | ----------------------- | --------------- | ---------- | --------------------------------------------------------------------- | ----------------------------------------------------------- |
| finetune_1677246816    | YOLOv4-CSP              | 50.4            | -          | `magalhaes/vision/checkpoints/finetune_1677246816/weights/best.pt`    | `magalhaes/vision/onnx/yolov4-csp_schreder.onnx`            |
| yolov7_1685970525      | YOLOv7                  | 52.0            | 48.5       | `magalhaes/vision/checkpoints/yolov7_1686143097/weights/best.pt`      | `magalhaes/vision/onnx/yolov7_schreder.onnx`                |
| yolov7_tiny_1686142923 | YOLOv7-tiny             | 43.5            | 40.8       | `magalhaes/vision/checkpoints/yolov7_tiny_1686142923/weights/best.pt` | `magalhaes/vision/onnx/yolov7-tiny_schreder.onnx`           |
| yolov7_tiny_1686851375 | YOLOv7-tiny (distilled) | 44.6            | 41.1       | `magalhaes/vision/checkpoints/yolov7_tiny_1686851375/weights/best.pt` | `magalhaes/vision/onnx/yolov7_tiny_distilled_schreder.onnx` |

## Train

**YOLOv4-CSP**
```bash
cd yolo
python train.py --weights /media/magalhaes/vision/pretrained/yolov4-csp.weights --cfg ./models/yolov4-csp-schreder.cfg --data ./data/schreder.yaml --project train/yolov4 --batch-size 10

OR

cd yolo/run
./finetune.sh ./data/schreder.yaml
```

**YOLOv7**
```bash
cd yolov7
python train.py --workers 8 --batch-size 64 --data ./data/schreder.yaml --cfg ./cfg/training/yolov7-schreder.yaml --weights /media/magalhaes/vision/pretrained/yolov7.pt --name $exp_name --hyp data/hyp.schreder.finetuning.yaml

OR

cd yolov7/run
./train_yolov7.sh ./data/schreder.yaml
```

**YOLOv7 tiny**

```bash
cd yolov7
python train.py --workers 8 --batch-size 64 --data ./data/schreder.yaml --cfg ./cfg/training/yolov7-tiny-schreder.yaml --weights /media/magalhaes/vision/pretrained/yolov7-tiny.pt --name $exp_name --hyp data/hyp.schreder.finetuning.tiny.yaml

OR

cd yolov7/run
./train_tiny.sh ./data/schreder.yaml
```

## Test

**YOLOv4-CSP**
```bash
python test.py --weights <checkpoint_path> --data ./data/schreder.yaml --batch-size 14 --conf-thres 0.05 --iou-thres 0.55  --device 0 --cfg ./models/yolov4-csp-schreder.cfg --names ./data/schreder.names
```

**YOLOv7**
```bash
cd yolov7
python test.py --weights <checkpoint_path> --data ./data/schreder.yaml --batch-size 14 --conf-thres 0.05 --iou-thres 0.55  --device 0 --cfg ./models/yolov4-csp-schreder.cfg --names ./data/schreder.names

OR

cd yolov7/run
./test.sh <checkpoint_path> ./data/schreder.yaml
```

### Generate detections

**YOLOv4-CSP**
```bash
cd yolo
python detect.py --weights /media/magalhaes/vision/checkpoints/finetune_1677246816/weights/best.pt --source "/media/magalhaes/schreder/images/raw*" --output /media/magalhaes/vision/outputs/ --conf-thres 0.05 --iou-thres 0.55 --device 0 --cfg models/yolov4-csp-schreder.cfg --names data/schreder.names --save-txt
```

**YOLOv7**

```bash
cd yolov7
python detect.py --output_dir /media/magalhaes/schreder/detections --conf-thres 0.05 --iou-thres 0.55 --source /media/magalhaes/schreder/images/ --device 0 --weights /media/magalhaes/vision/checkpoints/yolov7_1686143097/weights/best.pt --save-txt --save-conf
```

### Export ONNX

**YOLOv7**
```bash
cd yolov7/
python export.py yolov7-tiny.pt --grid --end2end --topk-all 100 --iou-thres 0.55 --conf-thres 0.1 --img-size 640 640 --onnx --dynamic-batch
```
