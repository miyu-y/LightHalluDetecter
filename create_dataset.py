import json
import random

with open("raw_rag_truth_dataset/response.jsonl", "r") as f:
    res_data = [json.loads(l) for l in f.readlines()]
with open("raw_rag_truth_dataset/source_info.jsonl", "r") as f:
    info_data = [json.loads(l) for l in f.readlines()]

id_list = [d["source_id"] for d in info_data]

random.seed(42)

id_train = [d["source_id"] for d in res_data if d["split"] == "train"]

id_train_qa = [d["source_id"] for d in info_data if d["task_type"] == "QA" and d["source_id"] in id_train]
id_train_d2t = [d["source_id"] for d in info_data if d["task_type"] == "Data2txt" and d["source_id"] in id_train]
id_train_sum = [d["source_id"] for d in info_data if d["task_type"] == "Summary" and d["source_id"] in id_train]


id_dev = random.sample(id_train_qa, 70) + random.sample(id_train_d2t, 70) + random.sample(id_train_sum, 70)

train_data = []
dev_data = []
test_data = []
for id in id_list:
    # anchor (input)
    for d in info_data[:]:
        if d["source_id"] == id:
            task_type = d["task_type"]
            source = d["source"]
            ref = d["source_info"]
            info_data.remove(d)
            break
    # text (output)
    num = 0
    for d2 in res_data[:]:
        if num == 6:
            break
        if d2["source_id"] == id:
            num += 1
            model_name = d2["model"]
            text = d2["response"]
            labels = 0 if d2["labels"] == [] else 1  # include hallucination → 1
            data = {
                "ref": str(ref),
                "text": text,
                "labels": labels,
                "source": source,
                "model": model_name,
                "task_type": task_type,
                "source_id": id,
            }
            if d2["split"] == "train":
                if id in id_dev:
                    dev_data.append(data)
                else:
                    train_data.append(data)
            else:
                test_data.append(data)
            res_data.remove(d2)

tri_test_data = []
# filter test data for triplet method
for i in range(len(test_data) // 6):
    score = 0
    for j in range(6):
        score += test_data[i * 6 + j]["labels"]
    # if all outputs are hallucination (score == 6) or no hallucination (score == 0), skip
    if score != 0 and score != 6:
        for k in range(6):
            tri_test_data.append(test_data[i * 6 + k])


with open("dataset/2rag_truth_train.json", "w") as f:
    json.dump(train_data, f, indent=4)
with open("dataset/2rag_truth_dev.json", "w") as f:
    json.dump(dev_data, f, indent=4)
with open("dataset/2rag_truth_test.json", "w") as f:
    json.dump(tri_test_data, f, indent=4)
#######
with open("raw_rag_truth_dataset/response.jsonl", "r") as f:
    res_data = [json.loads(l) for l in f.readlines()]
with open("raw_rag_truth_dataset/source_info.jsonl", "r") as f:
    info_data = [json.loads(l) for l in f.readlines()]

# span
test_data = []
for id in id_list:
    # document
    for d in info_data[:]:
        if d["source_id"] == id:
            task_type = d["task_type"]
            source = d["source"]
            ref = d["source_info"]
            info_data.remove(d)
            break
    # text
    num = 0
    for d2 in res_data[:]:
        if num == 6:
            break
        if d2["source_id"] == id:
            num += 1
            model_name = d2["model"]
            text = d2["response"]
            labels = 0 if d2["labels"] == [] else 1
            data = {
                "ref": str(ref),
                "text": text,
                "labels": labels,
                "hallucination_id": d2["labels"],  # location
                "hallucination": {"hallucination_list": [x["text"] for x in d2["labels"]]},
                "source": source,  # name of dataset
                "model": model_name,
                "task_type": task_type,
                "source_id": id,
            }
            if d2["split"] == "test":
                test_data.append(data)
            res_data.remove(d2)

tri_test_data = []
# filter test data for triplet method
for i in range(len(test_data) // 6):
    score = 0
    for j in range(6):
        score += test_data[i * 6 + j]["labels"]
    # if all outputs are hallucination (score == 6) or no hallucination (score == 0), skip
    if score != 0 and score != 6:
        for k in range(6):
            tri_test_data.append(test_data[i * 6 + k])
with open("dataset/2rag_truth_span_test.json", "w") as f:
    json.dump(tri_test_data, f, indent=4)
