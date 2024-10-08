# Forest health project
Welcome to the forests health monitoring project. The goal of the project is to be able to detect signs of an unhealthy forest based on satellite images.

## Overview
<ins>The project come in three stages of development :</ins>

1. First stage : Classify and segement patterns of deforestation based on a research paper on the Borneo forest
2. Second stage : Redo of the first itteration but in pytorch instead of tensorflow. Add a new vision transformer based model for both classificationa and segmentation.
3. Third stage : Classify and segment pattern of unhealthy forest applied to middle lattitude forests.

This repository represent the second stage of devlopment of the project.

<ins>To see the other stages :</ins>

1. [First stage of development](https://github.com/DridrM/forests_health_monitoring)

## Structure of the project
<ins>The project is divided in two main modules under the root module `fhealth` :</ins>

- [dataset](https://github.com/DridrM/forest_health_project/tree/main/src/fhealth/dataset) :
  - extract_data : Connect to GCP bucket, extract raw csv and image data.
  - transform_data : Transform image data to create features and labels ready to be ingested by deep learning model.
  - load_data : Wrap up extraction and transformation, store the csv and image data in cache (local folder path structure inside params.py).

- [modeling](https://github.com/DridrM/forest_health_project/tree/main/src/fhealth/modeling) : More to come

## Tools used
<ins>Here the main tools used in the principal modules :</ins>

> **WARNING**: This is not a exhaustive description.

- dataset :
  - Pydantic : Conception of the DataHandler objects
  - google.cloud : Connexion to google cloud storage service

- modeling :
  - Pytorch : Conception of the deep learning models

## How to use the project
> **More to come**

## Credits
<ins>Dataset and project idea (stage 1 and 2) :</ins>

- [ForestNet: Classifying Drivers of Deforestation in Indonesia using Deep Learning on Satellite Imagery](https://stanfordmlgroup.github.io/projects/forestnet/)

![Tests](https://github.com/DridrM/forest_health_project/actions/workflows/run_tests.yaml/badge.svg)
