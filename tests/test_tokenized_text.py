from fastcoref import FCoref, LingMessCoref
import spacy

texts = ['We are so happy to see you using our coref package. This package is very fast!',
         'The man tried to put the boot on his foot but it was too small.',
         "I have a dog. The dog's toys are really cool."]

model = FCoref(device='cpu')
preds = model.predict(texts=texts)
for p in preds:
    print(p.get_clusters())


texts = [["We", "are", "so", "happy", "to", "see", "you", "using", "our", "coref", "package", ".", "This", "package", "is", "very", "fast", "!"],
["The", "man", "tried", "to", "put", "the", "boot", "on", "his", "foot", "but", "it", "was", "too", "small", "."],
["I", "have", "a", "dog", ".", "The", "dog", "\'s", "toys", "are", "really", "cool", "."]]

model = FCoref(device='cpu')
preds = model.predict(texts, is_split_into_words=True)
for p in preds:
    print(p.get_clusters())

model = FCoref(device='cpu')
preds = model.predict(texts[0], is_split_into_words=True)
print(preds)
