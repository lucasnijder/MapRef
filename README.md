# Beyond Equivalence: The Role of Domain Knowledge in Ontology Mapping Refinement
Last update: 12-2-2024

## Overview
This repository contains the code for mapping refinement 

The current implementation takes pairs and predicts the semantic relation between them (exactMatch, narrowMatch, broadMatch) via relation prediction. 

NOTE: the ESCO-CompetentNL use case is not included as CompetentNL is not publicly available yet. 

## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Usage](#usage)
6. [Contributing](#contributing)
7. [License](#license)

---

## Installation
```python
pip install -r requirements.txt
```

---

## Preparing Test Cases
To prepare data sets for specific test cases, the main.py script can be used. With one command all steps for data preprocessing are executed. 

For preparing all data sets in the project (only relation prediction):
```
python3 main.py --exp data_all
```

For the ESCO-O*NET use case:
```
python3 main.py --exp data_ESCO-ONET
```

For the web use cases:
```
python3 main.py --exp data_stroma
```

For the biomedical use cases:
```
python3 main.py --exp data_bio
```

## Individual Data Set Preparation
To prepare individual data sets, the data_preparation.py script can be used directly. The raw datasets are located in the `./data/raw` folder. To prepare raw data run the command. The preprocessing is performed in two steps. The first step proccesses the raw data files  into triples in csv format. The second step processes the triple csvs into pkl files which can be loaded in the PyTorch DataSet objects. 

```
python3 data_preparation.py --action XXX
```
For example:
```
python3 data_preparation.py --action ESCO-CNL_mapping_to_triples
```

The XXX can be replaced by either raw to triple commands:

`ONET_ontology_to_triples`, convert raw O*NET file to triples scv
`ESCO_ontology_to_triples`, convert raw ESCO file to triples csv
`bio_ontologies_to_triples`, convert bio ontologies to triples csv
`stroma_ontologies_to_triples`, convert web ontologies to triples csv

`ESCO-ONET_mapping_to_triples`, convert raw ESCO-O*NET mapping to triples csv
`bio_mappings_to_triples`, convert raw biomedical mappings to csv
`stroma_mappings_to_triples`, convert web mappings to csv

Or by triples to train and test format commands:
- `train_set_converter`, convert train triples into the format required for training (.pkl)
- `test_set_converter`, convert test triples into the format reqruired for testing (.pkl)
For both commands you also need to specify `data_sets`, `task_type` and `balanced` to indicate the use case, task type (RP/TC) and balanced to undersample (T/F)

The train or test set is saved as two pickle (.pkl) files and a csv. The first pickle file contains the pairs/triples, the second pickle file contains the labels and the csv file contains pairs/triples and the labels for an inspectable overview. They are saved in either `./data/processed/train/default` or `./data/processed/test/default` and are named after the task and dataset. So for example: 
- `TC_ESCO-ONET_test_labels.pkl`
- `TC_ESCO-ONET_test_triples.pkl`
- `TC_ESCO-ONET_test.csv`
---

## Training
To train the model, run:

```python
python3 train.py --classifier_variant XXX --task_type XXX --train_set_names XXX,YYY,ZZZ --train_set_balanced XXX --anchors XXX --num_labels XXX --lr XXX --epochs XXX --batch_size XXX --parallelize XXX --balance_class_weights XXX
```
For example:
```
python train.py --classifier_variant distilBERT --task_type RP --train_set_names ESCO,ONET --train_set_balanced F --anchors default --num_labels 3 --lr 0.00001 --epochs 10 --batch_size 16 --parallelize F --balance_class_weights T
```

The variables are:
- `model_variant`: which model to use, e.g. dwsunimannheim/TaSeR, distilbert-base-uncased
- `classifier_variant`: which classifier to use, e.g. taser or distilbert 
- `task_type`: RP or TC
- `train_set_names`: name of the train set to be used. The name exists of the included datasets divided by a comma. E.g. if you prepared a train set with ESCO and CNL triples, you enter ESCO,CNL for this variable.
- `train_set_balanced`: whether you want to use a train set of which the classes have been balanced: T/F
- `anchors`: whether you want to use data with or without anchors. 
- `num_labels`: represents the number of classes the model has to predict. If the task_type is TC this is always 2, as there is a positive and a negative class. If the task type is RP this variable depends on the number of classes.
- `lr`: the learning rate (we use 0.00001)
- `epochs`: the number of epochs (we use 1)
- `batch_size`: the batch size (we use 16)
- `parallelize`: whether the training calculations should be paralellized over multiple GPUs
- `balance_class_weights`: whether you want to add class weights to the training in order to take class imbalances into account

Each model is given a name plus three integers, e.g. `holly136`. The training progress of the model can be found on Wandb.ai. The model is saved in the `./models` folder, e.g. for `holly136`: `./models/distilbert-base-uncased/holly136_weights.pth`

---

## Evaluation

To evaluate the model, run:

```python
python3 evaluate.py --model_variant XXX --classifier_variant XXX --eval_type XXX --dataset_name XXX --anchors XXX --task_type XXX --model_name XXX --num_labels XXX --taser_default_model XXX
```
For example:
```
python3 evaluate.py --model_variant distilbert-base-uncased --classifier_variant distilbert --eval_type test --dataset_name ESCO-ONET --anchors default --task_type RP --model_name philip475 --num_labels 3 --taser_default_model F
```

The variables are:
- `model_variant`: the base model of the trained model, e.g. `GroNLP/bert-base-dutch-cased`
- `classifier_variant`: which classifier was used on top of BERT's embeddings: default or demir 
- `eval_type`: which dataset type to use for evaluation, always use `test`
- `dataset_name`: which dataset to use for evaluation 
- `anchors`: whether to use a use case with anchors or not, always use `default`
- `task_type`: which task type to use (`RP` or `TC`)
- `model_name`: the name given to the model, e.g. `holly136`.
- `num_labels`: the number of classes the model can predict. Again, for TC this is 2 and for RP it depends on the number of classes. 
- `taser_default_model`: whether the TaSeR base model was used, if unknown use `F`
