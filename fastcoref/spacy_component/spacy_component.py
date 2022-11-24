from spacy import Language, util
from spacy.tokens import Doc, Span
from fastcoref import FCoref, LingMessCoref
from typing import List, Tuple


@Language.factory(
    "fastcoref",
    assigns=["doc._.resolved_text", "doc._.coref_clusters"],
    default_config={
        "model_architecture": 'FCoref',  # FCoref or LingMessCoref
        "model_path": 'biu-nlp/f-coref',  # You can specify your own trained model path
        "device": None,  # "cuda" or "cpu" None defaults to cuda
        "max_tokens_in_batch": 10000,
        "enable_progress_bar": True
    },
)
class FastCorefResolver:
    """a class that implements the logic from
    https://towardsdatascience.com/how-to-make-an-effective-coreference-resolution-model-55875d2b5f19"""
    def __init__(
            self,
            nlp,
            name,
            model_architecture: str,
            model_path: str,
            device,
            max_tokens_in_batch: int,
            enable_progress_bar,
    ):
        assert model_architecture in ["FCoref", "LingMessCoref"]
        if model_architecture == "FCoref":
            self.coref_model = FCoref(model_name_or_path=model_path, device=device, nlp=nlp, enable_progress_bar=enable_progress_bar)
        elif model_architecture == "LingMessCoref":
            self.coref_model = LingMessCoref(model_name_or_path=model_path, device=device, nlp=nlp, enable_progress_bar=enable_progress_bar)
        self.max_tokens_in_batch = max_tokens_in_batch
        # Register custom extension on the Doc
        if not Doc.has_extension("resolved_text"):
            Doc.set_extension("resolved_text", default="")
        if not Doc.has_extension("coref_clusters"):
            Doc.set_extension("coref_clusters", default=None)

    def _get_span_noun_indices(self, doc: Doc, cluster: List[Tuple]) -> List[int]:
        """
        > Get the indices of the spans in the cluster that contain at least one noun or proper noun
        :param doc: Doc
        :param cluster: List[Tuple]
        :return: A list of indices of spans that contain at least one noun or proper noun.
        """
        spans = [doc.char_span(span[0],span[1]) for span in cluster]
        spans_pos = [[token.pos_ for token in span] for span in spans]
        span_noun_indices = [
            i for i, span_pos in enumerate(spans_pos) if any(pos in span_pos for pos in ["NOUN", "PROPN"])
        ]
        return span_noun_indices

    def _get_cluster_head(self, doc: Doc, cluster: List[Tuple], noun_indices: List[int]):
        """
        > Given a spaCy Doc, a coreference cluster, and a list of noun indices, return the head span and its start and end
        indices
        :param doc: the spaCy Doc object
        :type doc: Doc
        :param cluster: a list of Tuples, where each tuple is a char indices of token in the document
        :type cluster: List[Tuple]
        :param noun_indices: a list of indices of the nouns in the cluster
        :type noun_indices: List[int]
        :return: The head span and the start and end indices of the head span.
        """
        head_idx = noun_indices[0]
        head_start,head_end = cluster[head_idx]
        head_span = doc.char_span(head_start,head_end)
        return head_span, [head_start, head_end]

    def _is_containing_other_spans(self,span: List[int], all_spans: List[List[int]]):
        """
        It returns True if there is any span in all_spans that is contained within span and is not equal to span
        :param span: the span we're checking to see if it contains other spans
        :type span: List[int]
        :param all_spans: a list of all the spans in the document
        :type all_spans: List[List[int]]
        :return: A list of all spans that are not contained in any other span.
        """
        return any([s[0] >= span[0] and s[1] <= span[1] and s != span for s in all_spans])

    def _core_logic_part(self,document: Doc, coref: List[int], resolved: List[str], mention_span: Span):
        """
        If the last token of the mention is a possessive pronoun, then add an apostrophe and an s to the mention.
        Otherwise, just add the last token to the mention
        :param document: Doc object
        :type document: Doc
        :param coref: List[int]
        :param resolved: list of strings, where each string is a token in the sentence
        :param mention_span: The span of the mention that we want to replace
        :return: The resolved list is being returned.
        """
        char_span = document.char_span(coref[0],coref[1])
        final_token = char_span[-1]
        final_token_tag = str(final_token.tag_).lower()
        test_token_test = False
        for option in ["PRP$", "POS", "BEZ"]:
            if option.lower() in final_token_tag:
                test_token_test = True
                break
        if test_token_test:
            resolved[char_span.start] = mention_span.text + "'s" + final_token.whitespace_
        else:
            resolved[char_span.start] = mention_span.text + final_token.whitespace_
        for i in range(char_span.start + 1, char_span.end):
            resolved[i] = ""
        return resolved

    def __call__(self, doc: Doc, resolve_text=False) -> Doc:
        """
        The function takes a doc object and returns a doc object
        :param doc: Doc
        :type doc: Doc
        :return: A Doc object with the resolved text and coreference clusters added as attributes.
        """
        preds = self.coref_model.predict(texts=[doc.text])
        clusters = preds[0].get_clusters(as_strings=False)
        if resolve_text:
            resolved = list(tok.text_with_ws for tok in doc)
            all_spans = [span for cluster in clusters for span in cluster]
            for cluster in clusters:
                indices = self._get_span_noun_indices(doc,cluster)
                if indices:
                    mention_span, mention = self._get_cluster_head(doc, cluster, indices)
                    for coref in cluster:
                        if coref != mention and not self._is_containing_other_spans(coref, all_spans):
                            self._core_logic_part(doc, coref, resolved, mention_span)
            doc._.resolved_text = "".join(resolved)
        doc._.coref_clusters = clusters
        return doc

    def pipe(self, stream, batch_size=512, resolve_text=False):
        for docs in util.minibatch(stream, size=batch_size):
            preds = self.coref_model.predict(
                    texts=[doc.text for doc in docs],max_tokens_in_batch=self.max_tokens_in_batch)
            for idx,pred in enumerate(preds):
                clusters = pred.get_clusters(as_strings=False)
                doc = docs[idx] 
                if resolve_text:    
                    resolved = list(tok.text_with_ws for tok in doc)
                    all_spans = [span for cluster in clusters for span in cluster]
                    for cluster in clusters:
                            indices = self._get_span_noun_indices(doc,cluster)
                            if indices:
                                mention_span, mention = self._get_cluster_head(doc, cluster, indices)
                                for coref in cluster:
                                    if coref != mention and not self._is_containing_other_spans(coref, all_spans):
                                        self._core_logic_part(doc, coref, resolved, mention_span)
                    doc._.resolved_text = "".join(resolved)
                doc._.coref_clusters = clusters
                yield doc