import argparse
import json
import evaluate
import numpy as np
from transformers import RobertaTokenizerFast, RobertaModel, DataCollatorForTokenClassification, AutoModelForTokenClassification, TrainingArguments, Trainer

#  Create labels to support begin of label and inside label
def modify_labels(labels_file):
    modified_labels2ids = {}
    modified_ids2labels = {}
    with open(labels_file, mode="r", encoding="utf-8") as lab_file:
        labels = json.load(lab_file)

    index = 1
    for _, label in labels.items():
        label_begin = "B-" + label
        label_inside = "I-" + label
        modified_labels2ids[index] = label_begin
        modified_labels2ids[index + len(labels)] = label_inside
        modified_ids2labels[label_begin] = index
        modified_ids2labels[label_inside] = index +len(labels)
        index += 1
    modified_labels2ids[0] = "O"
    modified_ids2labels["O"] = 0

    return modified_labels2ids, modified_ids2labels


def tokenize_and_align_labels(tokens, labels, tokenizer):
    new_labels = []
    new_tokens = []
    tokenized_input = tokenizer(tokens)
    # new_tokens = tokenizer.convert_ids_to_tokens(tokenized_input["input_ids"])
    # for i, label in enumerate(labels):
    #     word_ids = tokenized_input.word_ids(batch_index=i)
    #     previous_word_idx = None
    #     label_ids = []
    #     for word_idx in word_ids:
    #         if word_idx is None:
    #             new_labels.append(-100)
    #         elif word_idx != previous_word_idx:
    #             new_labels.append(label[word_idx])
    #         else:
    #             label_ids.append(-100)

    new_labels.append(-100)
    new_tokens.append('<s>')
    for i,token in enumerate(tokenized_input["input_ids"]):
        toks = tokenizer.convert_ids_to_tokens(token)
        for j,tok in enumerate(toks[1:len(toks)-1]):
            new_tokens.append(tok)
            if j == 0:
                new_labels.append(labels[i])
            else:
                new_labels.append(-100)

    new_labels.append(-100)
    new_tokens.append('</s>')

    return new_tokens, new_labels

# def compute_metrics(p):
#     predictions, labels = p
#
#     predictions = np.argmax(predictions, axis=2)
#     true_predictions = [[label_list[p] for (p,l) in zip(prediction,label) if l != -100] for prediction, label in zip(predictions, labels)]
#     true_labels = [
#         [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     results = seqeval.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }


def train(dataset_file, labels_file):
    new_labels2ids, new_ids2labels = modify_labels(labels_file)
    #  load
    tokenizer = RobertaTokenizerFast.from_pretrained("osiria/roberta-base-italian")
    with open(dataset_file, mode="r", encoding="utf-8") as data_file:
        data = json.load(data_file)

    for doc in data["train"]:
        words = doc["words"]
        labels = doc["labels"]
        doc["words"], doc["labels"] = tokenize_and_align_labels(words, labels, tokenizer)


    # words, labels = tokenize_and_align_labels("La bambina giocava benissimo a calcio e batteva tutti i suoi avversari.", [1,1,2,3,4,4,5,2,1,1,1,1], tokenizer)
    num_labels = len(new_labels2ids)
    model = AutoModelForTokenClassification.from_pretrained("osiria/roberta-base-italian", num_labels =num_labels, id2label=new_ids2labels, label2id=new_labels2ids)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    seqeval = evaluate.load("seqeval")



    training_args = TrainingArguments(
        output_dir="fgsd_model",
        learning_rate= 2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args = training_args,
        train_dataset= data["train"],
        eval_dataset= data["train"],
        tokenizer = tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics

    )

def main():
    parser = argparse.ArgumentParser(description="Pass labels and dataset")
    parser.add_argument("--dataset", type=str, help="Specify the path to your dataset in json format")
    parser.add_argument("--labels", type=str, help="Specify path to your ids to labels json file")
    args = parser.parse_args()
    train(args.dataset, args.labels)



if __name__ == "__main__":
    main()
