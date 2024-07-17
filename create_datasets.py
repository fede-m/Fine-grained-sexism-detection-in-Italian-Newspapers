import argparse
from collections import defaultdict
import json
import numpy as np
from sklearn.model_selection import StratifiedKFold
# from datasets import Dataset


def get_texts(file_name:str):
    ''' Converts jsonl file into list of default dictionaries containing text and entities. Each entity is also a dictionary
    @:param The JSONL file name/path
    @:return A dictionary with The structure of the final object is:
        {
        "text" : str,
        "entities": [
            {
                "id": int,
                "label": string,
                "start_offset": int,
                "end_offset": int
            }
        ]

    '''

    id2docs = defaultdict()
    doc_id = 0
    with open(file_name, mode="r", encoding="utf-8") as jsonl_f:
        for line in jsonl_f:
            curr_doc = defaultdict()
            curr_line = json.loads(line)
            curr_doc["text"] = curr_line["text"]
            curr_doc["entities"] = curr_line["entities"]

            id2docs[doc_id] = curr_doc
            doc_id += 1

    return id2docs


def get_labels_ids(labels_json: str):
    with open(labels_json, mode="r", encoding="utf-8") as label_j:
        labels_list = json.load(label_j)

    label2id = {label["text"]: idx +1 for idx, label in enumerate(labels_list)}
    label2id["NA"] = 0
    id2label = {idx +1: label["text"] for idx, label in enumerate(labels_list)}
    id2label[0] = "NA"

    return id2label,label2id

def create_k_fold_dataset(id2docs, labels2ids):

    final_docs = []
    spans_labels = []
    spans_groups = []

    #  Extracts spans, labels and create groups
    for i,doc in id2docs.items():

        #  Add the id of the label corresponding to each span
        if doc["entities"]:
            final_docs.extend([i for label in doc["entities"]])
            spans_labels.extend([labels2ids[label["label"]] for label in doc["entities"]])
            #  Add the id of the document where the span is present
            spans_groups.extend([i for label in doc["entities"]])
        else:
            final_docs.append(i)
            spans_labels.append(0)
            spans_groups.append(i)

    spans_labels_np = np.array(spans_labels)
    spans_groups_np = np.array(spans_groups)
    docs_np = np.array(final_docs)

#   Initialize sklearn StratifiedGroupKFold
    sgkf = StratifiedKFold(n_splits=10)
    print(sgkf.get_n_splits(docs_np, spans_labels_np))
    k_folded_dataset = defaultdict()

    for i, (train_index, test_index) in enumerate(sgkf.split(docs_np, spans_labels_np,spans_groups_np)):
        k_folded_dataset["Fold"+str(i)] = {"Train":[], "Test":[]}
        train = set()
        test = set()
        for index in train_index:
            train.add(spans_groups[index])

        k_folded_dataset["Fold" + str(i)]["Train"].extend([id2docs[ind] for ind in train])
        # kFoldedDataset["Fold" + str(i)]["Train"] = list(set(kFoldedDataset["Fold" + str(i)]["Train"]))

        for indx in test_index:
            test.add(spans_groups[indx])

        k_folded_dataset["Fold" + str(i)]["Test"].extend([id2docs[ind] for ind in test])

    count_labels_per_fold(k_folded_dataset, 10)

    return k_folded_dataset


def count_labels_per_fold(kFoldedDataset, k):

    for i in range(k):
        print("------ Fold"+str(i)+"------")
        labels_count_train = defaultdict(int)
        labels_count_test = defaultdict(int)

        print("Training length: ")
        for doc_train in kFoldedDataset["Fold"+str(i)]["Train"]:

            for label in doc_train["entities"]:
                labels_count_train[label["label"]] += 1

        print("Train labels distribution: ")
        print(labels_count_train)

        for doc_test in kFoldedDataset["Fold"+str(i)]["Test"]:
            for label in doc_test["entities"]:
                labels_count_test[label["label"]] += 1

        print("Test labels distribution: ")
        print(labels_count_test)

# def upload_Hugging_Face(k_folded_dataset):
#
#     for i, v in enumerate(k_folded_dataset.values()):
#         hf_dataset = Dataset.from_list(v["Train"])
#         hf_dataset.push_to_hub("fede-m/fine_grained_sexism_detection_italian_fold_"+str(i)+"_train", private=True)
#         hf_dataset = Dataset.from_list(v["Test"])
#         hf_dataset.push_to_hub("fede-m/fine_grained_sexism_detection_italian_fold_" + str(i) + "_test", private=True)

def store_folds(k_folded_dataset):
    for i, v in enumerate(k_folded_dataset.values()):
        with open("fold"+str(i)+".jsonl", mode="w", encoding="utf-8") as out_file:
            out_file.write(json.dumps(v, ensure_ascii=False))

def main():
    parser = argparse.ArgumentParser(description="Pass JSONL file path")
    parser.add_argument("--file_path", type=str, help="Specify the filepath of the JSONL file you want to pre-process")
    parser.add_argument("--labels", type=str, help="Specify the filepath of the JSON file containing the labels")
    args = parser.parse_args()

    id2docs = get_texts(args.file_path)
    ids2labels, labels2ids = get_labels_ids(args.labels)
    kFoldeddataset = create_k_fold_dataset(id2docs, labels2ids)

    store_folds(kFoldeddataset)


if __name__ == "__main__":
    main()