import pandas as pd
import uuid
import json
from metrics import CorefEvaluator
from util import flatten, extract_mentions_to_clusters


def gen_custom_data_tokens():
    df = pd.read_json('prepare_ontonotes/test.english.jsonlines', lines=True)
    df = df[['doc_key', 'sentences']]
    df = df.iloc[0:5]
    df['tokens'] = df['sentences'].apply(lambda x: flatten(x))
    df = df[['doc_key', 'tokens']]

    df.to_json('custom_data.jsonlines', orient='records', lines=True)


def gen_custom_data_raw():
    examples = [
        {'doc_key': uuid.uuid4().hex, 'text': 'The man tried to put the boot on his foot but it was too small.', },
        {'doc_key': uuid.uuid4().hex, 'text': 'Some apologizing was needed in the relationship after the argument because it is soothing.', }
    ]
    with open('custom_data_raw.jsonlines', 'w') as f:
        for doc in examples:
            f.write(json.dumps(doc) + "\n")


def test_clusters_output_file():
    df1 = pd.read_json('prepare_ontonotes/test.english.jsonlines', lines=True)
    df2 = pd.read_json('evaluation/test.english.output.jsonlines', lines=True)

    df1 = df1[['doc_key', 'clusters']]
    df2 = df2[['doc_key', 'clusters']]

    df = df1.merge(df2, on=['doc_key'])
    df['clusters_x'] = df['clusters_x'].apply(lambda clusters: tuple(tuple(tuple(m) for m in cluster) for cluster in clusters))
    df['clusters_y'] = df['clusters_y'].apply(lambda clusters: tuple(tuple(tuple(m) for m in cluster) for cluster in clusters))

    eval = CorefEvaluator()
    for i in df.index:
        gold_clusters = df.iloc[i]['clusters_x']
        predicted_clusters = df.iloc[i]['clusters_y']

        mention_to_gold_clusters = extract_mentions_to_clusters(gold_clusters)
        mention_to_predicted_clusters = extract_mentions_to_clusters(predicted_clusters)

        eval.update(predicted_clusters, gold_clusters, mention_to_predicted_clusters, mention_to_gold_clusters)

    print(eval.get_prf())


if __name__ == '__main__':
    # test_clusters_output_file()
    gen_custom_data_tokens()
    gen_custom_data_raw()

