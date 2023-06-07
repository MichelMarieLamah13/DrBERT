import itertools

import numpy as np
from seqeval.metrics import classification_report

from Model import trainer, output_name, test_tokenized_datasets, label_list


class TrainEvaluate:

    @staticmethod
    def process():
        trainer.train()
        trainer.evaluate()
        trainer.save_model(f"BEST-{output_name}")

        predictions, labels, _ = trainer.predict(test_tokenized_datasets)
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        f1_score = classification_report(
            list(itertools.chain.from_iterable(true_labels)),
            list(itertools.chain.from_iterable(true_predictions)),
            digits=4,
        )
        print(f1_score)
