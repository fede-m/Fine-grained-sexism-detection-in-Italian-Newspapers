The current project is organized as follows.

Folder "dataset" contains all datasets employed in this work, both the raw and the pre-processed one. In particular, it contains:
1) dataset.jsonl: contains the raw JSONL file we downloaded from doccano containing the original articles from the Webz.io dataset and the relevant annotation
2) folder baseline: contains the 5 cross-validation folds we used to train the baseline model. For each article we have only the text, labels and id. The same dataset can also be found on HuggingFace at "fede-m/fine_grained_sexism_detection_in_italian_newspapers_folds_FINAL".
3) folder pipeline: contains the 5 cross-validation folds we used to train the pipeline model. The text has already been sentence tokenized and co-reference resolution applied. For each sentence we have the text, the label for the binary classification, and the entities for the token classification. The same dataset can also be found on HuggingFace at "fede-m/setfit_dataset_coreference_folds".

Folder "labels" contains:
1) labels.json: contains the raw labels downloaded from doccano.
2) id2labels.json: contains the mapping between label ids and labels

Here, an overview of the main functionalities of the Python files and their relevance in the task.
1) partition_dataset.py : contains the code to partition the dataset into k-folds. The output corresponds to the dataset that can be found at "dataset/baseline". Note that the partition has been done once at the beginning both for the baseline and pipeline.
2) coreference.ipynb: Colab Notebook, contains the code to prepare the dataset for the pipeline. For each document in each fold coreferences are extracted, the text is sentence tokenized, and corresponding coreferences are added at the beginning of each sentence.
3) baseline_train.ipybn: Colab Notebook, contains the code to pre-process the documents (tokenization, truncation, padding, labels alignment) and train the baseline.
4) RoBERTa_binary.ipybn: Colab Notebook, contains the code to apply the RoBERTa model to the binary sentence classification task (sexism detection).
5) Setfit_binary.ipybn: Colab Notebook, contains the code to apply SetFit Binary and Multi-Class settings to the binary sentence classification task (sexism detection).
6) pipeline_train.ipybn: Colab Notebook, contains the complete pipeline with SetFit used for inference pre-filtering the training and test datasets to feed the RoBERTa token classifier.
