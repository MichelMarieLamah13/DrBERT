import itertools

from datasets import load_dataset
from seqeval.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForTokenClassification

import uuid
import evaluate
from transformers import DataCollatorForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback

import numpy as np

# Chargement des données
dataset = load_dataset("Dr-BERT/QUAERO", "medline")

print(dataset)

# Chargement du model
tokenizer = AutoTokenizer.from_pretrained("Dr-BERT/DrBERT-7GB")

task = "ner"


def tokenize_and_align_labels(examples):
    """
    Methode pour traitement des données
    :param examples: les données
    :return:
    """
    label_all_tokens = True
    tokenized_inputs = tokenizer(list(examples["tokens"]), truncation=True, is_split_into_words=True, max_length=512)

    labels = []

    for i, label in enumerate(examples[f"{task}_tags"]):

        label_ids = []
        previous_word_idx = None
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])

            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)

            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


# Traitement des données
train_dataset = dataset["train"]
train_tokenized_datasets = train_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)

dev_dataset = dataset["validation"]
dev_tokenized_datasets = dev_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)

test_dataset = dataset["test"]
test_tokenized_datasets = test_dataset.map(tokenize_and_align_labels, batched=True, keep_in_memory=True)

label_list = train_dataset.features["ner_tags"].feature.names
model = AutoModelForTokenClassification.from_pretrained("Dr-BERT/DrBERT-7GB", num_labels=len(label_list))


def getConfig(raw_labels):
    label2id = {}
    id2label = {}

    for i, class_name in enumerate(raw_labels):
        label2id[class_name] = str(i)
        id2label[str(i)] = class_name

    return label2id, id2label


label2id, id2label = getConfig(label_list)
model.config.label2id = label2id
model.config.id2label = id2label

output_name = f"DrBERT-QUAERO-{task}"

args = TrainingArguments(
    output_name,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    load_best_model_at_end=True,
    greater_is_better=True,
)

metric = evaluate.load("seqeval", experiment_id=output_name)
data_collator = DataCollatorForTokenClassification(tokenizer)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_predictions = [[label_list[p] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                        zip(predictions, labels)]
    true_labels = [[label_list[l] for (p, l) in zip(prediction, label) if l != -100] for prediction, label in
                   zip(predictions, labels)]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": results["overall_precision"], "recall": results["overall_recall"], "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"]}


trainer = Trainer(
    model,
    args,
    train_dataset=train_tokenized_datasets,
    eval_dataset=dev_tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
