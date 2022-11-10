from fastcoref import FCoref, LingMessCoref
import spacy

model = FCoref(device='cpu')

texts = ['We are so happy to see you using our coref package. This package is very fast!',
         'The man tried to put the boot on his foot but it was too small.',
         "I have a dog. The dog's toys are really cool."]

preds = model.predict(texts=texts)
for p in preds:
    print(p.get_clusters())

preds = model.predict(texts[0])
print(preds)

texts = [["We", "are", "so", "happy", "to", "see", "you", "using", "our", "coref", "package", ".", "This", "package", "is", "very", "fast", "!"],
["The", "man", "tried", "to", "put", "the", "boot", "on", "his", "foot", "but", "it", "was", "too", "small", "."],
["I", "have", "a", "dog", ".", "The", "dog", "\'s", "toys", "are", "really", "cool", "."]]

preds = model.predict(texts, is_split_into_words=True)
for p in preds:
    print(p.get_clusters())

preds = model.predict(texts[0], is_split_into_words=True)
print(preds)

texts = ["cant fool you", ["or", "can", "I", "?"]]
model = FCoref(device='cpu')
preds = model.predict(texts, is_split_into_words=False)
print(preds)
