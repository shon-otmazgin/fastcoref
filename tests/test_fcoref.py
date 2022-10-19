import unittest

from fastcoref import FCoref, CorefResult


class TestFCoref(unittest.TestCase):
    def setUp(self) -> None:
        self.test_text = ['We are so happy to see you using our coref package. This package is very fast!',
                          'The man tried to put the boot on his foot but it was too small.']

        self.model = FCoref()
        self.expected_clusters = [[[(0, 2), (33, 36)], [(33, 50), (52, 64)]],
                                  [[(0, 7), (33, 36)], [(33, 41), (46, 48)]]]
        self.expected_clusters_strings = [[['We', 'our'], ['our coref package', 'This package']],
                                          [['The man', 'his'], ['his foot', 'it']]]

    def test_predict_with_unexpected_object(self):
        texts = {'text1': 'sss'}
        with self.assertRaises(ValueError) as exc:
            preds = self.model.predict(texts=texts)
        self.assertEqual(str(exc.exception), f"texts argument expected to be a list of strings, "
                                             f"or one single text string. provided {type(texts)}")

    def test_predict_with_single_string(self):
        preds = self.model.predict(texts=self.test_text[0])

        self.assertIsInstance(preds, CorefResult)

    def test_predict_with_list(self):
        preds = self.model.predict(texts=self.test_text)

        self.assertIsInstance(preds, list)
        for res_obj in preds:
            self.assertIsInstance(res_obj, CorefResult)

    def test_get_clusters(self):
        preds = self.model.predict(texts=self.test_text)

        self.assertIsInstance(preds, list)
        for i, res_obj in enumerate(preds):
            self.assertIsInstance(res_obj, CorefResult)
            self.assertListEqual(res_obj.get_clusters(as_strings=True), self.expected_clusters_strings[i])

    def test_get_clusters_indices(self):
        preds = self.model.predict(texts=self.test_text)

        self.assertIsInstance(preds, list)
        for i, res_obj in enumerate(preds):
            self.assertIsInstance(res_obj, CorefResult)
            self.assertListEqual(res_obj.get_clusters(as_strings=False), self.expected_clusters[i])

    def test_get_logits(self):
        preds = self.model.predict(texts=self.test_text)
        self.assertIsInstance(preds, list)

        self.assertGreater(preds[0].get_logit(span_i=(33, 50), span_j=(52, 64)), 0)
        self.assertGreater(preds[1].get_logit(span_i=(21, 29), span_j=(46, 48)), 0)
