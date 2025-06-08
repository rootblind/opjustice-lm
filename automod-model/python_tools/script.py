import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optoolkit import CNNTransformerClassifier, CNNModelTrainer, DatasetLoader
from transformers import AutoTokenizer
import torch
from datasets import Dataset
from imblearn.over_sampling import RandomOverSampler
import numpy as np

dataset = DatasetLoader(["Message"], "Message")

train = dataset.dataset["train"]
x_train = np.array(dataset.dataset["train"]["Message"]).reshape(-1, 1)

columns = dataset.labels
labels = np.stack([train[col] for col in columns], axis=1)

label_keys = ["".join(map(str, row)) for row in labels.tolist()]

ros = RandomOverSampler()
texts_resampled, label_keys_resampled = ros.fit_resample(x_train, label_keys)

labels_resampled = [list(map(int, list(key))) for key in label_keys_resampled]

resampled_train = Dataset.from_dict({
    "Message": [x[0] for x in texts_resampled],
    **{col: [label[i] for label in labels_resampled] for i, col in enumerate(columns)}
})

print(resampled_train)
print(dataset.dataset["train"])