from collections import defaultdict
import json
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def get_texts(file_name:str):
    ''' Converts jsonl file into list of default dictionaries containing text and entities. Each entity is also a dictionary
    @:param The path to the JSONL file that contains the dataset
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
            curr_doc["words"] = curr_line["text"]
            curr_doc["labels"] = curr_line["entities"]
            curr_doc["id"] = doc_id

            id2docs[doc_id] = curr_doc
            doc_id += 1

    return id2docs

    
def get_labels_ids(labels_json: str):
    """
    Function to parse labels from a JSON file and create mappings between labels and their IDs.

    @:param labels_json: The path to the JSONL file that contains the labels

    @:return two dictionaries:
        - id2label: a mapping from label IDs to their corresponding text 
        - label2id: a mapping from label text to their corresponding ID
    """
    with open(labels_json, mode="r", encoding="utf-8") as label_j:
        labels_list = json.load(label_j)

    label2id = {label["text"]: idx +1 for idx, label in enumerate(labels_list)}
    # Label "NA" is added for tokens that do not have a label
    label2id["NA"] = 0
    id2label = {idx +1: label["text"] for idx, label in enumerate(labels_list)}
    id2label[0] = "NA"

    return id2label,label2id


def create_k_fold_dataset(id2docs, labels2ids):
    """
    Function to create a k-fold dataset for cross-validation purposes.
    
    @:param id2docs: A dictionary mapping ids to documents.
    @:param labels2ids: A dictionary mapping labels to ids.
    
    @:return A list of lists containing the k-fold dataset.
    """
    final_docs = []
    spans_labels = []
    spans_groups = []

    #  Extracts spans, labels and create groups
    for i, doc in id2docs.items():

        #  Add the id of the label corresponding to each span
        if doc["labels"]:
            final_docs.extend([i for _ in doc["labels"]])
            spans_labels.extend([labels2ids[label["label"]] for label in doc["labels"]])
            #  Add the id of the document where the span is present
            spans_groups.extend([i for _ in doc["labels"]])
        else:
            final_docs.append(i)
            spans_labels.append(0)
            spans_groups.append(i)

    spans_labels_np = np.array(spans_labels)
    spans_groups_np = np.array(spans_groups)
    docs_np = np.array(final_docs)

#   Initialize sklearn StratifiedKFold
    sgkf = StratifiedGroupKFold(n_splits=5)

    k_folded_dataset = [] # List that will contain the k-fold dataset with the ids of the documents in each partition
    for i, (_, test_index) in enumerate(sgkf.split(docs_np, spans_labels_np,spans_groups_np)):
        # set that will contain the ids of the documents that belong to the current fold
        test = set()

        # Iterate over the test index, which contains the ids of the documents in the current partition
        for indx in test_index:
            # Add the id of the document to the test set
            test.add(spans_groups[indx])

        # Add the test set to the k_folded_dataset list
        k_folded_dataset.append(list(test))

    check_document_occurring_multiple_times(k_folded_dataset)
    count_labels_per_fold(k_folded_dataset,5, id2docs)

    return k_folded_dataset

def check_document_occurring_multiple_times(folded_dataset):
    """
    Function to check if a document occurs in multiple folds in a k-fold dataset.

    @:param folded_dataset: A list of lists, where each inner list contains the ids of the documents in a partition of the dataset.

    The function raises a ValueError with a message indicating the document id and the number of times it occurs in the dataset.
    """

    document_presence = defaultdict(int)

    # Loop through the documents in each fold and check if documents occurr multiple times
    for i in range(len(folded_dataset)):
        for j in range(len(folded_dataset[i])):
            curr_doc_id = folded_dataset[i][j]
            document_presence[curr_doc_id] += 1


    for k, v in document_presence.items():
        if v > 1:
            print("---- This document occurred " + str(v)+" times ----")
            print(k)


def count_labels_per_fold(k_folded_dataset, k, id2docs):
    """
    Function to count the number of occurrences of each label per fold in a k-fold dataset.

    @:param k_folded_dataset: A list of lists, where each inner list contains the ids of the documents in a partition of the dataset.
    @:param k: The number of folds in the k-fold dataset
    @:param id2docs: A dictionary mapping ids to documents.

    The function prints the label and the number of occurrences for each label per fold.
    """
    for j in range(k):
        labels_count_train = defaultdict(int)
        labels_count_test = defaultdict(int)
        for i in range(k):
            if i == j:
                print("------ Fold"+str(j)+" as validation ------")
                for doc in k_folded_dataset[i]:
                    if id2docs[doc]["labels"]:
                        for label in id2docs[doc]["labels"]:
                            labels_count_test[label["label"]] += 1
                    else:
                        labels_count_test["NA"] += 1
            else:
                for doc_train in k_folded_dataset[i]:
                    if id2docs[doc_train]["labels"]:
                        for label in id2docs[doc_train]["labels"]:
                            labels_count_train[label["label"]] += 1
                    else:
                        labels_count_train["NA"] += 1

        print("Train labels distribution: ")
        print(labels_count_train)
        print("Test labels distribution: ")
        print(labels_count_test)


def main():

    dataset_filename = "dataset/dataset.jsonl"
    labels_filename = "labels/labels.json"
    id2label, label2id = get_labels_ids(labels_filename)
    id2doc = get_texts(dataset_filename)
    folds = create_k_fold_dataset(id2doc, label2id)

    # Generate the final folds by converting the ids of the documents in the dataset to the corresponding documents
    for i,fold in enumerate(folds):
        new_fold = {"Data": [id2doc[doc_id] for doc_id in fold]}
        # Store the generated folds in json files
        with open("dataset/baseline/fold0"+str(i)+".json", "w", encoding="utf-8") as outfile:
            json_fold = json.dumps(new_fold, ensure_ascii=False)
            outfile.write(json_fold)

if __name__ == "__main__":
    main()