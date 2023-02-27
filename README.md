# Deep Learning

This repository contains a framework for building and training deep learning models. The framework is built using
TensorFlow and is designed to be easy to use and extend. The main components of the framework are the Model and
DataProcessing classes, which handle the creation and training of models, as well as the loading and preprocessing of
datasets.

## Installation

Before using this repository, you will need to install TensorFlow and set up a virtual environment for running Jupyter
notebooks.

### Tensorflow

To install TensorFlow, follow the official [installation instructions](https://www.tensorflow.org/install/pip). Be sure
to meet the system requirements before installing.

### Virtual Environment for jupyter-notebooks in conda environment

Activate the conda environment created before for installing TensorFlow and install jupyter in that environment:

```
conda activate my-conda-env 
conda install jupyter       
jupyter-notebook
```

### Additional Packages

You will also need to install any additional packages required by the models and examples in this repository. These
packages include numpy, matplotlib, etc. They can be installed using pip or conda.

## Code Structure:

The Deep Learning Repository is organized into two main classes: **Model** and **DataProcessing**. The Model class is
responsible for defining, training, and evaluating the deep learning network. The DataProcessing class is responsible
for loading and processing the dataset.

Each deep learning network has its own config directory and Jupyter notebook. The config directory contains two YAML
files, one for dataset parameters and one for model parameters. The Jupyter notebook serves as a visual interface for
monitoring the progress and evaluations of the network.

The _data_processing_ directory contains the implementation of the DataProcessing class, which loads the dataset and
carries out necessary preprocessing steps. The dataset must be in the format provided by the Dataset-Rendering
repository, which includes a JSON file with all relevant information and several directories for different types of data
such as images and labels. However, real-world data can also be stored in this format. An example of how to do this can
be found in the _example_dataset_ subdirectory.

The models directory contains the Model class, which serves as the main interface for all network-related functions. The
structure of the network is defined in the _network.py_ file, and common building blocks such as the backbone of the
network are defined in the corresponding files in the _network_elements_ subdirectory. The main connection between the
Model and DataProcessing classes is provided by the two member dictionaries of the DataProcessing class:
_input_data_cfg_
and _output_data_cfg_.

The losses and metrics directories contain custom-defined loss and metric classes, respectively. The plots directory
contains all plotting scripts, and the utils directory contains additional features such as a parser for redefining
parameters set in the configuration file from the terminal.

## Config Files

### dataset.yaml

#### Path to the Dataset

The path to the dataset is defined by two variables: BASE_PATH_DATA and NAME. BASE_PATH_DATA defines the base directory
where the datasets are stored, and NAME defines the specific name of the dataset. The full path to the dataset is
created by combining these two variables: base_path + / + name

#### Input-Output Data Specification

The input output data specification of the network is defined as follows:

```
# mask encoding types: 0: INSTANCE, 1: CATEGORY, 2: BINARY
in:
  img:
    name: "in_img"
    shape: [320, 180]
  edge:
    name: "in_edge"
    mask_encoding: 2
    shape: [ 80, 45 ]
  contour: null
  segmentation: null
  prior_img: null

out:
  edge:
    name: "out_edge"
    mask_encoding: 2
    shape: [ 320, 180 ]
  contour: null
  segmentation: null
  flow: null
```

The "name" parameter is used as the signature name of the input and output Tensor. It is used to determine which loss
function should be applied to which output of the network. Additionally, it is also used in C++ to copy the input and
output data to the correct tensors of the model. For more information, please refer to the Tensorflow documentation.

The "mask_encoding" parameter defines how the ground truth data is applied in the network. The following options can be
set:

* Instance of object: 0, for each object another label
* Category: 1, category such as bar, sheet, base, tube, ...
* Binary: 2, edge or non edge

#### Individual Datasets Specification (Test, Train, Real world)

For each dataset type (Test, Train, and Real World) the data processing steps can be defined individually. The
parameters of each dataset type are represented by a dictionary with the same structure. The Real World dataset has an
additional parameter, PATH, which specifies its location. As an example, the dictionary for the Train dataset is
provided below.
```
TRAIN:
  NAME: Train
  MAX_IMG: 1500
  BATCH_SIZE: 3
  CACHE: true
  SHUFFLE: true
  PREFETCH: true
  DATA_AUGMENTATION:
    brightness: 0.25
    contrast_factor: 0.75
    hue: 0.05
    saturation: 0.75
    apply_random_flip: true
    noise_std: 0
    apply_spotlight: false
    spotlight_strength: 0.3 #max additional value and brightness
```

### model.yaml

The config file contains several parameters that are used to define the number of training iterations, the loss function to be applied, callback functions, the learning rate, etc. Some of these parameters need to match a specific implementation in the code. These parameters are explained in more detail below:

* The "name" parameter is used to define the name of the directory in which the model is stored, as well as the architecture of the model. Therefore, the given name needs to match one of the conditional statements in the get_model() function in the network.py file.
* The "loss" dictionary needs to match the structure defined in the get_loss_function() function in the model.py file. For example, certain variables may need to be defined for edge detection tasks, while for segmentation tasks, you may only need to set a null value or "true" if a segmentation loss should be applied.
* The "padding" parameter allows setting the number of pixels at the border of the image, for those the loss function is not computed.
* The "region of attraction variables set the number of adjacent pixels to the ground truth pixel with low loss"


## Applying the Model in the App

In order to use the model in the App, add the following to flag to the building plan and specify the name of the trained model

building Plan:
```
{
  "options": {
    "CNN": "LEPENet_Mock_Timber_Wall"
  },
},
```


Furthermore, copy the resulting tflite file to the assets folder of the viot_android repository. The tflite file is given here:
* path: model_cfg['BASE_PATH'] + dataset_name + model_cfg['NAME'] + 'TFLITE' + '*.tflite'
