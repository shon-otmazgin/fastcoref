from fastcoref import spacy_component
import spacy
texts = ['We are so happy to see you using our coref package. This package is very fast!',
         'Glad to see you are using the spacy component as well. As it is a new feature!',
         'The man tried to put the boot on his foot but it was too small.',
         'Alice goes down the rabbit hole. Where she would discover a new reality beyond her expectations.'
        ]

nlp_fcoref = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"]) #Resolving references requires pos tagging
nlp_fcoref.add_pipe("fastcoref",config={'model_name':'FCoref','device':'cuda:0'})
# Test __call__
doc = nlp_fcoref(texts[0])
print(doc._.resolved_text)
print(doc._.coref_clusters)
# Test pipe
doc_list = nlp_fcoref.pipe(texts)
for doc in doc_list:
    print(doc._.resolved_text)
    print(doc._.coref_clusters)


nlp_lingmess = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "ner", "textcat"])#Resolving references requires pos tagging
nlp_lingmess.add_pipe("fastcoref",config={'model_name':'LingMessCoref','device':'cuda:0'})
# Test __call__
doc = nlp_lingmess(texts[0])
print(doc._.resolved_text)
print(doc._.coref_clusters)
# Test pipe
doc_list = nlp_lingmess.pipe(texts)
for doc in doc_list:
    print(doc._.resolved_text)
    print(doc._.coref_clusters)