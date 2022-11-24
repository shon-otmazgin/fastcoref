from fastcoref import FCoref, LingMessCoref

texts = ['We are so happy to see you using our coref package. This package is very fast!',
         'The man tried to put the boot on his foot but it was too small.',
         "I have a dog. The dog's toys are really cool."]

model = FCoref(device='cpu')

preds = model.predict(texts=texts[0], max_tokens_in_batch=5000, output_file='out_test.jsonlines')
print(preds)

preds = model.predict(texts=texts, max_tokens_in_batch=5000, output_file='out_test.jsonlines')
print(preds)

model = FCoref(device='cpu', enable_progress_bar=False)

preds = model.predict(texts=texts, max_tokens_in_batch=5000, output_file='out_test.jsonlines')
print(preds)

for p in preds:
    print(p.get_clusters())
    print(p.get_clusters(as_strings=False))
print(preds[0].get_logit(span_i=(33, 50), span_j=(52, 64)))

model = LingMessCoref(device='cpu')
preds = model.predict(texts=texts, max_tokens_in_batch=5000)

for p in preds:
    print(p.get_clusters())
    print(p.get_clusters(as_strings=False))
print(preds[0].get_logit(span_i=(33, 50), span_j=(52, 64)))
print(preds[1].get_logit(span_i=(21, 29), span_j=(46, 48)))
