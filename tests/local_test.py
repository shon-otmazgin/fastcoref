from fastcoref import FCoref, LingMessCoref

texts = ['We are so happy to see you using our coref package. This package is very fast!',
         'The man tried to put the boot on his foot but it was too small.']

model = FCoref(device='cuda:0')
# preds = model.predict(texts={'text1': 'sss'}, max_tokens_in_batch=5000)
preds = model.predict(texts=texts[0], max_tokens_in_batch=5000)
print(preds)

preds = model.predict(texts=texts, max_tokens_in_batch=5000)
print(preds)

# preds = model.predict(texts=texts, max_tokens_in_batch=5000)
#
# print(preds[0].get_clusters())
# print(preds[0].get_clusters(string=True))
# print(preds[0].get_logit(span_i=(33, 50), span_j=(52, 64)))
# print(preds[1].get_clusters())
# print(preds[1].get_clusters(string=True))
#
# model = LingMessCoref(device='cuda:0')
# preds = model.predict(texts=texts, max_tokens_in_batch=5000)
#
# print(preds[0].get_clusters())
# print(preds[0].get_clusters(string=True))
# print(preds[0].get_logit(span_i=(33, 50), span_j=(52, 64)))
# print(preds[1].get_clusters())
# print(preds[1].get_clusters(string=True))
# print(preds[1].get_logit(span_i=(21, 29), span_j=(46, 48)))





