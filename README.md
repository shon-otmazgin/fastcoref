This repository is the official implementation of the paper ["F-COREF: Fast, Accurate and Easy to Use Coreference Resolution"](https://arxiv.org/abs/2209.04280).

The `fastcoref` Python package provides an easy and fast API for coreference information with only few lines of code without any prepossessing steps.

- [Installation](#Installation)
- [Demo](#demo)
- [Quick start](#quick-start)
- [Spacy component](#spacy-component)
- [Training](#distil-your-own-coref-model)
- [Citation](#citation)

## Installation

```python
pip install fastcoref
```

## Demo

**NEW** try out the FastCoref web demo

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/pythiccoder/FastCoref)

Credit: Thanks to @aribornstein !

## Quick start

The main functionally of the package is the `predict` function.
The return value of the function is a list of `CorefResult` objects, from which one can extract the coreference clusters (either as strings or as character indices over the original texts), as well as the logits for each corefering entity pair:

```python
from fastcoref import FCoref

model = FCoref(device='cuda:0')

preds = model.predict(
   texts=['We are so happy to see you using our coref package. This package is very fast!']
)

preds[0].get_clusters(as_strings=False)
> [[(0, 2), (33, 36)],
   [(33, 50), (52, 64)]
   ]

preds[0].get_clusters()
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
## Spacy component
The package also provides a custom Spacy component that can be plugged into a Spacy(V3) pipeline. The custom component can be added to your pipeline using the name "fastcoref" after importing spacy_component. You can choose the model_architecture as either FCoref or LingMessCoref. If you'd like to use your own trained version you can specify the model_path parameter pointing to your model. The example below shows how to use the pre-trained FCoref model.
```python
from fastcoref import spacy_component
texts = ['We are so happy to see you using our coref package. This package is very fast!',
         'Glad to see you are using the spacy component as well. As it is a new feature!'
        ]
nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])
nlp.add_pipe("fastcoref",config={'model_architecture':'FCoref',"model_path":'biu-nlp/f-coref','device':'cuda'})
```
After adding the model to your Spacy pipeline, the coreference clusters derived by the model can be accessed through the .\_.coref_clusters attribute of the outputted documents. It is also possible to access a version of the text with the coreferences already resolved through the .\_.resolved_text attribute. However this second attribute requires you to pass an optional resolved_text parameter as True to the component config when passing the documents through the pipeline. Important things to note are; Resolving the text requires the tagger component to be in the pipeline and is slower than just extracting the coreference clusters implementation-wise.
```python

doc_list = nlp.pipe(texts,component_cfg={"fastcoref":{'resolve_text':True}})
for doc in doc_list:
   print(doc._.resolved_text)
   print(doc._.coref_clusters)
```

## Distil your own coref model
On top of the provided models, the package also provides the ability to train and distill coreference models on your own data, opening the possibility for fast and accurate coreference models for additional languages and domains.

To be able to distil your own model you need:
1. A Large unlabeled dataset, for instance Wikipedia or any other source.
2. A teacher model to annotate clusters for this dataset. For instance, It can be the package `LingMess` model.
3. A student model, in the below example, we define the teacher model architecture, but you can modify it with another set of hyper-parameters. 

Dataset guidelines:
1. Each dataset split (train/dev/test) should be in separate file.
2. Each file should be in `jsonlines` format
3. Each json line in the file must include these fields:
   1. `doc_key` (you can use `uuid.uuid4().hex` to generate or any other keys)
   2. `tokens` field, if you have plain text, it is recommended to run [`Spacy`](https://spacy.io/) tokenizer to get tokens.
   3. `clusters` information as a span start/end indices of the `tokens`.

As mentioned before, you can have the clusters and the tokens information using this package to any unlabeled dataset. 

Once you done preparing annotate dataset files, you can do the following for training:
```
git clone https://github.com/shon-otmazgin/fastcoref.git
cd fastcoref/hard_training
```

```python
python run.py \
      --cache_dir=CACHE_DIR \
      --output_dir=DIR_NAME  \
      --overwrite_output_dir \
      --model_name_or_path=distilroberta-base \     # or any other you would like.
      --train_file=PATH_TO_TRAIN_FILE.jsonlines \
      --dev_file=PATH_TO_DEV_FILE.jsonlines \
      --test_file=PATH_TO_TEST_FILE.jsonlines \
      --max_tokens_in_batch=5000 \                  # configure based on your max length document and your GPU size.
      --do_train \
      --eval_split=dev \
      --logging_steps=500 \
      --eval_steps=1000 \
      --train_epochs=50 \
      --head_learning_rate=3e-5  \
      --learning_rate=1e-5 \
      --ffnn_size=1024 \                           # you can have larger coreference head with this parameter.
      --experiment_name="your-custom-fastcoref" \
      --device=cuda:0
```

After finish training your own model, push the model the huggingface hub (or keep it local), and load your model:
```python
from fastcoref import FCoref

model = FCoref(
   model_name_or_path='your-fast-coref-model-path',
   device='cuda:0'
)
```
Or in case of `LingMessCoref` model:
```python
model = LingMessCoref(
   model_name_or_path='your-fast-coref-model-path',
   device='cuda:0'
)
```


## Citation