from fastcoref import FCoref, LingMessCoref, CorefArgs

texts = ['We are so happy to see you using our coref package. This package is very fast!',
         'The man tried to put the boot on his foot but it was too small.']


# args = CorefArgs(
#     model_name_or_path='you-awesome-model',
#     device='cpu'
# )
# model = FCoref(args=args)


model = FCoref()

preds = model.predict(texts=texts, max_tokens_in_batch=5000)
print(preds)

print(preds[0].get_clusters())
print(preds[0].get_clusters(as_strings=False))
print(preds[0].get_logit(span_i=(33, 50), span_j=(52, 64)))
print(preds[1].get_clusters())
print(preds[1].get_clusters(as_strings=False))


model = LingMessCoref()
preds = model.predict(texts=texts, max_tokens_in_batch=5000)

print(preds[0].get_clusters())
print(preds[0].get_clusters(as_strings=False))
print(preds[0].get_logit(span_i=(33, 50), span_j=(52, 64)))
print(preds[1].get_clusters())
print(preds[1].get_clusters(as_strings=False))
print(preds[1].get_logit(span_i=(21, 29), span_j=(46, 48)))









