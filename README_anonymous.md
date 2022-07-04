# LingMess: Linguistically Informed Multi Expert Scorers for Coreference Resolution
This repository is the official implementation of the paper "LingMess: Linguistically Informed Multi Expert Scorers for Coreference Resolution"

Credit: Many code parts were taken from [s2e-coref](https://github.com/yuvalkirstain/s2e-coref#requirements) repo.

## Table of contents

- [Environments and Requirements](#environments-and-requirements)
- Create Datasets
   * [Prepare OntoNotes dataset](#prepare-ontonotes-dataset)
   
     OR
  
   * [Prepare your own custom dataset](#prepare-your-own-custom-dataset)
- [Training](#training)
- [Inference](#inference)


## Environments and Requirements

Below tested on `Ubuntu 20.04.3 LTS` with `Python 2.7` and `Python 3.7`
```
conda create -y --name py27 python=2.7
conda create -y --name lingmess-coref python=3.7 && conda activate lingmess-coref && pip install -r requirements.txt
```
Note: Python 2.7 is for OntoNotes dataset preprocess. 

## Create Datasets

### Prepare OntoNotes dataset

Download [OntoNotes 5.0](https://catalog.ldc.upenn.edu/LDC2013T19) corpus (registration is required).

Important: `ontonotes-release-5.0_LDC2013T19.tgz` must be under `prepare_ontonotes` folder.

Setup (~45 min):
```
cd prepare_ontonotes
chmod 755 setup.sh
./setup.sh
``` 
Credit: This script was taken from the [e2e-coref](https://github.com/kentonl/e2e-coref/) repo.

### Prepare your own custom dataset

Our implementation supports also custom dataset, both for training and inference.

Custom dataset guidelines:
1. Each dataset split (train/dev/test) should be in separate file.
2. Each file should be in `jsonlines` format
3. Each json line in the file must include these fields:
   1. `doc_key` (you can use `uuid.uuid4().hex` to generate)
   2. `text` or `tokens` field, if you choose to use `text` we will run [`Spacy`](https://spacy.io/) tokenizer.

option #1:
```
    {"doc_key": "DOC_KEY_1", "text": "this is document number 1, it's text is raw text"},
```   
option #2:
```
    {"doc_key": "DOC_KEY_2", "tokens": ["this", "is", "document", "number", "1", ",", "it", "'s", "text", "is", "tokenized"],
```

Note: Optional - speaker information by token.

Note: If you want to train the model on your own dataset, please provide `clusters` information as a span start/end indices of the `tokens` attribute.

## Training
Currently the implementation supports `['longformer', 'roberta', 'bert']` transformers, but it should be easy to use any other transformer.

Replicate Train on `OntoNotes` with `Longformer`
```
python run.py \
       --output_dir=lingmess-longformer \
       --overwrite_output_dir \
       --model_name_or_path=allenai/longformer-large-4096 \
       --train_file=prepare_ontonotes/train.english.jsonlines \
       --dev_file=prepare_ontonotes/dev.english.jsonlines \
       --test_file=prepare_ontonotes/test.english.jsonlines \
       --max_tokens_in_batch=5000 \
       --do_train \
       --eval_split=dev \
       --logging_steps=500 \
       --eval_steps=1000 \
       --train_epochs=129 \
       --head_learning_rate=3e-4 \
       --learning_rate=1e-5 \
       --ffnn_size=2048 \
       --experiment_name="lingmess" \
       --device=cuda:0
```

## Inference

Inference on you own dataset
```
python run.py \
       --model_name_or_path=PATH_TO_TRAINED_MODEL \ 
       --output_file=OUTPUT_FILE_PATH.jsonlines \
       --test_file=PATH_TO_FILE_TO_PREDICT.jsonlines \
       --eval_split=test \
       --max_tokens_in_batch=15000 \
       --device=cuda:0
```
The output file  located at `OUTPUT_FILE_PATH.jsonlines` 

The output file includes the predicted clusters. They are a list of coreference clusters such as each cluster contains a list of mention offsets `[start, end]` . These offsets correspond to the tokenization in the field `tokens`.


To Replicate Evaluation on `OntoNotes`, set `--test_file` to OntoNotes test file.
