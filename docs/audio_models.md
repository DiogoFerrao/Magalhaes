# Audio Models

In this page we describe the models available for multi-label audio classification and the commands necessary to train, test, and generate detections.

## Available Models

We have two types of models available, the distilled which are the models pre-trained using knowledge distillation, and the finetuned which are the models trained using the Schréder dataset. From this group only the finetuned support the set of classes of interest for this project. Jointly with the  finetuned models, we also provide their F1-score on the cross validation using two folds (conf_thres=0.5).

Since we are currently unable to export PyTorch's MelSpectrogram to ONNX ([issue 81075](https://github.com/pytorch/pytorch/issues/81075)), we have replaced it with a custom implementation based on [torchlibrosa implementation](https://github.com/qiuqiangkong/torchlibrosa/blob/master/torchlibrosa/stft.py). The models using this implementation have `_export` in their name and their performance is expected to be similar to the original models.

> **Note** the weights and the name of the finetuned models correspond to the model that was trained using the full dataset. The cross validation results correspond to identical models trained using a two fold split of the dataset.

**Distilled models**

| Name                                  | Model       | Weights                                                                         |
| ------------------------------------- | ----------- | ------------------------------------------------------------------------------- |
| online_export_yolov7_tiny_1684760021  | ResNet-tiny | `magalhaes/sound/distillations/online_export_yolov7_tiny_1684760021/last0.pth`  |
| online_teacher_yolov7_tiny_1680619925 | ResNet-tiny | `magalhaes/sound/distillations/online_teacher_yolov7_tiny_1680619925/last0.pth` |

**Finetuned models**

| Name                                               | Model                 | F1-score | Weights                                                                                           |
| -------------------------------------------------- | --------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| distilled_yolov7_tiny_export_full_train_1686124695 | ResNet-tiny-distilled | 0.663    | `magalhaes/sound/checkpoints/distilled_yolov7_tiny_export_full_train_1686124695/model_best_1.pth` |
| distilled_yolov7_tiny_1684787496                   | ResNet-tiny-distilled | 0.669    |                                                                                                   |
| yolov7_tiny_1684788651                             | ResNet-tiny           | 0.657    |                                                                                                   |
| yolov4_1684788694                                  | ResNet                | 0.667    |                                                                                                   |

## Train

**ResNet-tiny**

```bash
cd rethink
python train.py --config_path ./schreder_yolov7_tiny_distilled.json --exp_name resnet-tiny-distilled

OR

cd rethink/run
./train.sh ./schreder_yolov7_tiny_distilled.json resnet-tiny-distilled
```

## Test

**ResNet-tiny**

```bash
cd rethink
python evaluate.py --config_path ./schreder_yolov7_tiny_distilled.json
```

## Generate detections

**ResNet-tiny**

```bash
cd rethink
python detect.py <input_dir> --model yolov7_tiny --checkpoint <checkpoint_path> --out_dir <output_dir>
```

## Export to ONNX

```bash
cd rethink/models
python export.py <model-weights> --arch yolov7_tiny
```
