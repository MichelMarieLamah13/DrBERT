import os

from Model import output_name, task
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report


class Test:
    @staticmethod
    def test():
        tokenizer = AutoTokenizer.from_pretrained(f"BEST-{output_name}")
        model = AutoModelForTokenClassification.from_pretrained(f"BEST-{output_name}")
        nlp = pipeline(task, model=model, tokenizer=tokenizer, aggregation_strategy="simple")
        files = Test.get_files("CAS")
        nb_files = len(files)
        for i, filename in enumerate(files):
            basename, ext = os.path.splitext(filename)
            with open(f"CAS/{filename}") as file:
                txt = file.read()
                res = nlp(txt)
                df = pd.DataFrame(res)
                df.to_csv(f'ANN/{basename}.csv', index=False)
                print(f"[{i}/{nb_files}] -> File {filename} annotated")

    @staticmethod
    def metrics():
        files = Test.get_files("CORRECT")
        nb_files = len(files)
        true_labels, predicted_labels = [], []
        for i, filename in enumerate(files):
            basename, ext = os.path.splitext(filename)
            df = pd.read_csv(f"CORRECT/{filename}")
            true_labels.extend(df['correct_label'].values)
            predicted_labels.extend(df['entity_group'].values)
        precision = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)
        report = classification_report(true_labels, predicted_labels, zero_division=0)
        # Affichage des résultats
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print(f"Report: \n {report}")

    @staticmethod
    def read_corpora(name: str):
        with open(name) as f:
            return f.read()

    @staticmethod
    def get_files(folder_path: str):
        return [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
