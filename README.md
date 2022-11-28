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

if your text is already tokenized use `is_split_into_words=True`
```python
preds = model.predict(
   texts = [["We", "are", "so", "happy", "to", "see", "you", "using", "our", "coref", 
             "package", ".", "This", "package", "is", "very", "fast", "!"]],
   is_split_into_words=True
)
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

The package also provides a custom [SpaCy](https://spacy.io/) component that can be plugged into a Spacy(V3) pipeline. 
The example below shows how to use the pre-trained `FCoref` model.

```python
from fastcoref import spacy_component
import spacy


text = 'Alice goes down the rabbit hole. Where she would discover a new reality beyond her expectations.'

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("fastcoref")

doc = nlp(text)
doc._.coref_clusters
> [[(0, 5), (39, 42), (79, 82)]]
```

**Note**: it is better to `exclude=["parser", "lemmatizer", "ner", "textcat"]` at `spacy.load` since the component only rely on pos tagging.


You can also load other models, e.g. the more accurate model `LingMessCoref`:

```python
nlp.add_pipe(
   "fastcoref", 
   config={'model_architecture': 'LingMessCoref', 'model_path': 'biu-nlp/lingmess-coref', 'device': 'cpu'}
)
```

By specifying `resolve_text=True` in the pipe call, you can get the resolved text for each cluster:

```python
docs = nlp.pipe(
   texts, 
   component_cfg={"fastcoref": {'resolve_text': True}}
)

docs[0]._.resolved_text
> "Alice goes down the rabbit hole. Where Alice would discover a new reality beyond Alice's expectations."
```

## Distil your own coref model
On top of the provided models, the package also provides the ability to train and distill coreference models on your own data, opening the possibility for fast and accurate coreference models for additional languages and domains.

To be able to distil your own model you need:
1. A Large unlabeled dataset, for instance Wikipedia or any other source.

   Guidelines:
   1. Each dataset split (train/dev/test) should be in separate file.
      1. Each file should be in `jsonlines` format
      2. Each json line in the file must include at least one of:
         1. `text: str` - a raw text string.
         2. `tokens: List[str]` - a list of tokens (tokenized text).
         3. `sentences: List[List[str]]` - a list of lists of tokens (tokenized sentences).
      3. `clusters` information (see next for annotation) as a span start/end indices of the provided field `text`(char level) `tokens`(word level) `sentences`(word level)`.


2. A model to annotate the clusters. For instance, It can be the package `LingMessCoref` model.
```python
from fastcoref import LingMessCoref

model = LingMessCoref()
preds = model.predict(texts=texts, output_file='train_file_with_clusters.jsonlines')

```

3. Train and evaluate your own `FCoref`
```python
from fastcoref import TrainingArgs, CorefTrainer

args = TrainingArgs(
    output_dir='test-trainer',
    overwrite_output_dir=True,
    model_name_or_path='distilroberta-base',
    device='cuda:2',
    epochs=129,
    logging_steps=100,
    eval_steps=100
)   # you can control other arguments such as learning head and others.

trainer = CorefTrainer(
    args=args,
    train_file='train_file_with_clusters.jsonlines', 
    dev_file='path-to-dev-file',    # optional
    test_file='path-to-test-file'   # optional
)
trainer.train()
trainer.evaluate(test=True)

trainer.push_to_hub('your-fast-coref-model-path')

```

After finish training your own model, push the model the huggingface hub (or keep it local), and load your model:
```python
from fastcoref import FCoref

model = FCoref(
   model_name_or_path='your-fast-coref-model-path',
   device='cuda:0'
)
```


## Citation

```
@inproceedings{Otmazgin2022FcorefFA,
  title={F-coref: Fast, Accurate and Easy to Use Coreference Resolution},
  author={Shon Otmazgin and Arie Cattan and Yoav Goldberg},
  booktitle={AACL},
  year={2022}
}
```

[F-coref: Fast, Accurate and Easy to Use Coreference Resolution](https://aclanthology.org/2022.aacl-demo.6) (Otmazgin et al., AACL-IJCNLP 2022)