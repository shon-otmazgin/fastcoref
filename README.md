# Fastcoref

This repository is the official implementation of the paper ["F-COREF: Fast, Accurate and Easy to Use Coreference Resolution"](https://arxiv.org/abs/2209.04280).

The `fastcoref` Python package provides an easy and fast API for coreference information with only few lines of code without any prepossessing steps.

- [Installation](#installation)
- [Demo](#demo)
- [Quick start](#quick-start)
- [Spacy component](#spacy-component)
- [Training](#distil-your-own-coref-model)
- [Citation](#citation)

## Installation

```bash
pip install fastcoref
# or for training:
pip install fastcoref[train]
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
doc = nlp(      # for multiple texts use nlp.pipe
   text,
   component_cfg={"fastcoref": {'resolve_text': True}}
)

doc._.resolved_text
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

1. Train and evaluate your own `FCoref`

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
    test_file='path-to-test-file',   # optional
    nlp=nlp # optional, for custom nlp class from spacy
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

## Performance

For maximum inference throughput:

```python
from fastcoref import FCoref

model = FCoref(device='cuda:0', compile_model=True)

# First call triggers compilation (~6s one-time cost)
preds = model.predict(texts=['warm up text'])

# All subsequent calls run at full speed (~3ms per text)
preds = model.predict(texts=texts, max_tokens_in_batch=10000)
```

- Use `compile_model=True` for long-running processes (APIs, pipelines). The one-time 6s compilation cost amortizes quickly.
- Batch multiple texts in a single `predict()` call — per-text cost drops to ~0.6ms in batches of 10+.
- `max_tokens_in_batch` controls GPU memory usage. Higher values = faster (more parallelism) but more VRAM. Lower values = safer for limited GPU memory.
- Release logits after use in large-scale inference to free memory:

```python
for pred in preds:
    clusters = pred.get_clusters()
    pred.release_logits()  # free the logit matrix
```

## Changelog

### v2.2.0 — Performance & Stability

This release focuses on performance improvements and critical bug fixes without any changes to model accuracy.

**Critical Bug Fixes:**

- **Fixed batch size calculation that could cause OOM crashes (and system restarts).** The `DynamicBatchSampler` was computing effective batch length from the *shortest* example in a batch, but since the dataset is sorted ascending by length, all subsequent examples are longer. This caused actual GPU memory usage to far exceed `max_tokens_in_batch`, potentially crashing the GPU driver. Now uses the current (longest) example's length for the calculation.
- **Fixed memory leak in `CorefResult`.** Every prediction stored the full coref logits matrix (`[max_k, max_k+1]` float32) even when `get_logit()` was never called. For large-scale inference this accumulated hundreds of MB. Logits are now stored lazily and can be explicitly released via `result.release_logits()`.
- **Fixed `set_seed` fragile coupling.** The function assumed the passed object always had `n_gpu` attribute, which could fail silently.

**Performance Improvements:**

- **SDPA attention for FCoref (2-4x attention speedup).** FCoref uses RoBERTa which supports PyTorch's Scaled Dot-Product Attention. LingMess uses Longformer (sparse attention) which is architecturally incompatible with SDPA, so it continues using eager attention.
- **`torch.inference_mode()` replaces `torch.no_grad()`.** Faster inference by disabling autograd tracking AND tensor version counting (~5-10% speedup).
- **67x faster tokenization.** Spacy was running `tok2vec` and `attribute_ruler` pipeline components unnecessarily. The package only needs spacy's rule-based tokenizer for word splitting and char offsets. Now uses `nlp.tokenizer.pipe()` directly with all pipeline components excluded.
- **Optimized cluster label computation during training.** Replaced O(batch × k²) nested Python loops with cluster-based lookup approach using pre-built mention-to-index mappings.
- **80x faster predict() calls.** Removed HuggingFace `Dataset.from_dict()` + `.map()` overhead from the inference path. The framework was serializing the tokenizer via dill/pickle on every call, adding ~237ms of overhead to a 3ms operation. Now calls `encode()` directly with a lightweight dict-based dataset.
- **20x faster category label computation (LingMess training).** Replaced O(k²) Python loop with vectorized numpy broadcasting for pronoun categories (75% of pairs) and inverted index for entity-entity pairs (only checks pairs sharing at least one word).
- **`torch.compile` support for inference (3.8x forward speedup).** Pass `compile_model=True` to `FCoref()` or `LingMessCoref()`. First call pays a one-time ~6s compilation cost; all subsequent calls benefit from fused GPU kernels. Produces identical outputs to non-compiled.

**Modernization:**

- **Updated AMP API.** Replaced deprecated `torch.cuda.amp.GradScaler()` and `torch.cuda.amp.autocast()` with modern `torch.amp.GradScaler('cuda')` and `torch.amp.autocast('cuda')`.
- **Streaming batch creation for training.** Batches are collected as a list of dicts (no redundant tensor duplication) and shuffled by batch order rather than materializing a full HuggingFace Dataset of all batches.

**Notes:**

- All changes are backward-compatible. The public API (`predict`, `get_clusters`, `get_logit`) is unchanged.
- Spacy `en_core_web_sm` is still loaded for tokenization rules/vocab, but no pipeline components run. If you pass a custom `nlp` object (e.g. for the spacy component with POS tagging), it will be used as-is.

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
