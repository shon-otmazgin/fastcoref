from datasets import Dataset


d = Dataset.from_dict({"input_ids": []})

d = d.add_item({"input_ids": [1, 2, 3]})
d = d.add_item({"input_ids": [4, 5,]})
d = d.add_item({"input_ids": [6, 7, 80, 44]})

print(d)

print(d[0]["input_ids"])