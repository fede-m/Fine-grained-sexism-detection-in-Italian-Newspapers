'''
    Input: JSONL file containing the annotated texts from doccano
    Goal: map each label to an ID and then map each token to a label.
    Words with no assigned label will be given token label "0"
    Output: Pandas Dataframe with text and true labels for each word
'''

import argparse
from collections import defaultdict
import json
import numpy as np
import nltk


def get_texts(file_name:str):
    ''' Converts jsonl file into list of default dictionaries containing text and entities. Each entity is also a dictionary
    @:param The JSONL file name/path
    @:return The structure of the final object is:
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

    documents = []
    with open(file_name, mode="r", encoding="utf-8") as jsonl_f:
        for line in jsonl_f:
            curr_doc = defaultdict()
            curr_line = json.loads(line)
            curr_doc["text"] = curr_line["text"]
            curr_doc["entities"] = curr_line["entities"]
            documents.append(curr_doc)
    return documents


def get_tokens_labels(spans: list, tokens: list, labels: list, label2id: defaultdict):
    '''
    @:brief Start tokenizing the texts using nltk
    @:param
    @:return list of label names
    '''

    labels_per_token = [0]*len(spans)
    if not labels:
        return labels_per_token
    print(len(labels))
    for i, span in enumerate(spans):
        # print(tokens[i])
        # print(span)
        for label in labels:
            if label["start_offset"] <= span[0] <= span[1]<= label["end_offset"]:
                labels_per_token[i] = label2id[label["label"]]
                print(label["label"] + ": " + tokens[i])
                print("Span at 0: " + str(span[0]))
                print("Span at 0: " + str(span[1]))
                print("Start span: " + str(label["start_offset"]))
                print("End span: " + str(label["end_offset"]))
                break

    return labels_per_token


def get_labels_ids(labels_json: str):
    with open(labels_json, mode="r", encoding="utf-8") as label_j:
        labels_list = json.load(label_j)

    label2id = {label["text"]: idx +1 for idx, label in enumerate(labels_list)}
    label2id["NA"] = 0
    id2label = {idx +1: label["text"] for idx, label in enumerate(labels_list)}
    id2label[0] = "NA"

    return id2label,label2id

def pre_processing(file_name:str, labels_json:str):
    #  Get documents and labels
    docs = get_texts(file_name)
    ids2labels, labels2ids = get_labels_ids(labels_json)

    #  Create Dataset using K-Fold validation

    final = {"train": []}
    wp_tok = nltk.WordPunctTokenizer()
    for doc in docs:
        final_obj = defaultdict()
        final_obj["words"] = wp_tok.tokenize(doc["text"])
        spans = list(wp_tok.span_tokenize(doc["text"]))
        final_obj["labels"] = get_tokens_labels(spans, final_obj["words"], doc["entities"], labels2ids)
        final["train"].append(final_obj)

    return final, ids2labels


def main():
    parser = argparse.ArgumentParser(description = "Pass JSONL file path")
    parser.add_argument("--file_path", type=str, help="Specify the filepath of the JSONL file you want to pre-process")
    parser.add_argument("--labels", type=str, help="Specify the filepath of the JSON file containing the labels")
    args = parser.parse_args()

    training_data, ids2labels = pre_processing(args.file_path, args.labels)

    #  Upload the model to Hugging-Face


    #  Or store it locally
    with open("output.json", mode="w", encoding="utf-8") as output:
        json.dump(training_data,output, ensure_ascii=False)

    with open("id2labels.json", mode = "w", encoding="utf-8") as labels_file:
        json.dump(ids2labels, labels_file, ensure_ascii=False)

if __name__ == "__main__":
    main()




