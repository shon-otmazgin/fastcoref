The `fastcoref` Python package provides an easy and fast API for coreference information with only few lines of code without any prepossessing steps.

- [Installation](#Installation)
- [Quick start](#quick-start)
- [Training](#training)
- [Citation](#citation)

## Installation

```python
pip install fastcoref
```

## Quick start

The main functionally of the package is the `predict` function.
The return value of the function is a list of `CorefResult` objects, from which one can extract the coreference clusters (either as strings or as character indices over the original texts), as well as the logits for each corefering entity pair:

```python
from fastcoref import FCoref

model = FCoref(device='cuda:0')

preds = model.predict(
    texts=['We are so happy to see you using our coref package. This package is very fast!']
)

preds[0].get_clusters()
> [[(0, 2), (33, 36)], 
   [(33, 50), (52, 64)]
  ]

preds[0].get_clusters(string=True)
> [['We', 'our'], 
   ['our coref package', 'This package']
  ]
 
preds[0].get_logit(
    span_i=(33, 50), span_j=(52, 64)
)

> 18.852894
```

Processing can be applied to a collection of texts of any length in a batched and parallel fashion:

```python
texts = ['text 1', 'text 2',.., 'text n']

# control the batch size 
# with max_tokens_in_batch parameter

preds = model.predict(
    texts=texts, max_tokens_in_batch=100
)
```

The `max_tokens_in_batch` parameter can be used to control the speed vs. memory consumption (as well as speed vs. accuracy) tradeoff, and can be tuned to maximize the utilization of the associated hardware.

Lastly,
To use the larger but more accurate [`LingMess`](https://huggingface.co/biu-nlp/lingmess-coref) model, simply import `LingMessCoref` instead of [`FCoref`](https://huggingface.co/biu-nlp/f-coref):

```python
from fastcoref import LingMessCoref

model = LingMessCoref(device='cuda:0')
```

## Training
On top of the provided models, the package also provides the ability to train and distill coreference models on your own data, opening the possibility for fast and accurate coreference models for additional languages and domains.

Custom dataset guidelines:
1. Each dataset split (train/dev/test) should be in separate file.
2. Each file should be in `jsonlines` format
3. Each json line in the file must include these fields:
   1. `doc_key` (you can use `uuid.uuid4().hex` to generate or any other keys)
   2. `tokens` field, if you have plain text, it is recommended to run [`Spacy`](https://spacy.io/) tokenizer to get tokens.
   3. `clusters` information as a span start/end indices of the `tokens` attribute.

Then you can do the following for training:
```
git clone https://github.com/shon-otmazgin/fastcoref.git
cd fastcoref/hard_training
```

```python
python run.py \
      --cache_dir=CACHE_DIR \
      --output_dir=DIR_NAME  \
      --overwrite_output_dir \
      --model_name_or_path=distilroberta-base \
      --train_file=PATH_TO_TRAIN_FILE.jsonlines \
      --dev_file=PATH_TO_DEV_FILE.jsonlines \
      --test_file=PATH_TO_TEST_FILE.jsonlines \
      --max_tokens_in_batch=5000 \
      --do_train \
      --eval_split=dev \
      --logging_steps=500 \
      --eval_steps=1000 \
      --train_epochs=50 \
      --head_learning_rate=3e-5  \
      --learning_rate=1e-5 \
      --ffnn_size=1024 \
      --top_lambda=0.25 \
      --experiment_name="your-custom-fastcoref" \
      --device=cuda:7
```

## Citation