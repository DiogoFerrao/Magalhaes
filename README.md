# Magalhaes

**IMPORTANT** - the code assumes that the data is stored in the `/media/magalhaes` directory, please change the paths accordingly.

Project collectively developed by the team at INESC-ID.

Includes the work done for the thesis of Diogo Ferrão: Data Augmentation for Urban Environmental Sound Classification on Edge Devices.

## Setup

```
conda env create -f environment.yml
```

or

```
pip install -r requirements.txt
```

## Vision

### Models

* [Available Models](docs/vision_models.md#available-models)
* [Train](docs/vision_models.md#train)
* [Test](docs/vision_models.md#test)
* [Generate detections](docs/vision_models.md#generate-detections)
* [Export to ONNX](docs/vision_models.md#export-onnx)

### Data

* [Available Datasets](docs/vision_data.md#available-datasets)
* [Data Engine](docs/vision_data.md#data-engine)
* [Add new data](docs/vision_data.md#add-new-data)
* [Create dataset](docs/vision_data.md#create-dataset)
* [Search dataset](docs/vision_data.md#search-dataset)
* [Detect data leaks](docs/vision_data.md#detect-data-leaks)

## Sound

### Models

* [Available Models](docs/audio_models.md#available-models)
* [Train](docs/audio_models.md#train)
* [Test](docs/audio_models.md#test)
* [Generate detections](docs/audio_models.md#generate-detections)
* [Export to ONNX](docs/audio_models.md#export-to-onnx)

### Data

* [Data sources](docs/audio_data.md#data-sources)
* [Available Datasets](docs/audio_data.md#available-datasets)
* [Data Engine](docs/audio_data.md#data-engine)
* [Add new data](docs/audio_data.md#add-new-data)
* [Create dataset](docs/audio_data.md#create-dataset)
* [Preprocess audio](docs/audio_data.md#preprocess-audio)
* [Search dataset](docs/audio_data.md#search-dataset)

## Media structure

We also provide a more detailed description of the media structure in `/media/magalhaes/README.md`

```
/media/magalhaes
├── DAWN
│   ├── Sand
│   ├── Snow
│   ├── Fog
│   │   ├── images
│   │   └── labels
│   └── Rain
│       ├── images
│       └── labels
├── ESC-50
│   └── audio
├── ExDark
│   ├── images
│   └── labels
├── FSD50K
│   └── dev_audio
├── coco
│   ├── images
│   │   └── val2017
│   └── labels
│       └── val2017
├── emergency_vehicles
│   ├── images
│   └── labels
├── schreder
│   ├── labels
│   └── images
├── schreder_sound
│   ├── 15_12_2022
│   │   └── ground_truth
│   ├── 16_12_2022
│   │   └── ground_truth
│   ├── 18_12_2022
│   │   └── ground_truth
│   ├── 21_12_2022
│   │   └── ground_truth
│   ├── 22_9_2022
│   |   └── ground_truth
│   ...
│   └── outdoor_silence
├── sound
│   ├── checkpoints
│   ├── datasets
│   ├── onnx
│   ├── pretrained
│   └── spectograms
└── vision
    ├── checkpoints
    ├── datasets
    ├── onnx
    └── pretrained
```
