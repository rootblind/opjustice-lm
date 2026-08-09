# Architecture


## 1. System Purpose

`opjustice-lm` is a Romanian-language, multi-label text-classification system intended to score Discord messages against moderation/toxicity labels.

The supplied model description defines these labels:

- `OK`
- `Aggro`
- `Violence`
- `Sexual`
- `Hateful`

The model is based on `readerbench/RoBERT-base` and is oriented toward the moderation levels of the League of Legends Romania Discord server.

The implementation uses Hugging Face Transformers, Hugging Face Datasets, PyTorch, scikit-learn, imbalanced-learn, pandas, NumPy, `regex`, and `nlpaug`.

## 2. High-Level Architecture

```text
                    ┌─────────────────────────┐
                    │ Raw / Hugging Face Data │
                    │ CSV or saved Dataset    │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │     DatasetLoader       │
                    │ load / labels / encode  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      DataToolkit        │
                    │ clean / split / augment │
                    │ / resample / transform  │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Hugging Face Tokenizer  │
                    │ max_length = 512        │
                    └────────────┬────────────┘
                                 │
                                 ▼
             ┌──────────────────────────────────────┐
             │             Model Layer              │
             │                                      │
             │  Model                              │
             │  └─ AutoModelForSequenceClassification
             │                                      │
             │  CNNTransformerClassifier            │
             │  └─ Transformer → Conv1d → Pool → FC│
             └──────────────────┬───────────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │       Trainer Layer     │
                    │ ModelTrainer / CNNModel │
                    │ Trainer / CustomTrainer │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Saved Model Version     │
                    │ model_versions/vN       │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │      Inference          │
                    │ probabilities → labels  │
                    └─────────────────────────┘
```

## 3. Source Files and Responsibilities

### `optoolkit.py`

The supplied `optoolkit.py` is the central toolkit module. It contains the following logical components.

### `DatasetLoader`

Responsibilities:

- Load a dataset from a local Hugging Face `save_to_disk` directory when available.
- Otherwise load the configured Hugging Face dataset and save it locally.
- Identify label columns by excluding configured non-label columns.
- Build `id2label` and `label2id` mappings.
- Tokenize text and construct a multi-label matrix.
- Convert datasets to PyTorch format.
- Expose raw test text and labels through `Xy_test`.

The default local dataset path is `./automod-model/dataset`, and the default Hugging Face dataset name is `rootblind/opjustice-dataset`.

### `Model`

Responsibilities:

- Load a Hugging Face sequence-classification model.
- Select CUDA when available, otherwise CPU.
- Configure the model as `multi_label_classification`.
- Perform inference.
- Convert sigmoid probabilities to binary predictions.
- Convert binary predictions to label names.
- Apply the special `OK` rule: if `OK` is predicted, the returned label list becomes `['OK']`.
- Apply a dataset row-by-row labeling operation through `label_dataset`.

Inference tokenization uses truncation and a maximum length of 512.

### `Evaluator`

Responsibilities:

- Store test inputs, expected labels, and a tokenizer.
- Generate predictions for the test split.
- Calculate F1, precision, recall, accuracy, and ROC AUC.
- Generate a multi-label confusion matrix.

Supported averaging modes are passed through to scikit-learn.

### `DataToolkit`

This is the data-manipulation layer. It currently combines several distinct concerns:

- Translation.
- Duplicate removal.
- Label-based row filtering.
- Text normalization.
- Dataset filtering.
- Shuffling.
- Back-translation augmentation.
- `nlpaug` word augmentation.
- Train/test splitting.
- Tokenization.
- Random oversampling.
- SMOTE-Tomek.
- ADASYN resampling.

### `RegexClassifier` and `DataRegexClassifier`

These components provide a deterministic rule-based classification path.

`RegexClassifier`:

1. Receives trigger words.
2. Generates regex patterns.
3. Searches lower-cased messages.
4. Returns a Boolean match.

`DataRegexClassifier` applies the classifier to a DataFrame and creates a single label column.

### `ModelTrainer`

Wraps Hugging Face `Trainer` for the standard transformer classifier.

It:

- Stores model/tokenizer/datasets/training configuration.
- Computes class weights.
- Overrides loss using `BCEWithLogitsLoss`.
- Computes multi-label F1, ROC AUC, and accuracy.
- Initializes Hugging Face `TrainingArguments`.
- Trains, evaluates, and saves the model.

Observed defaults include:

- learning rate: `2e-5`
- batch size: `16`
- epochs: `8`
- weight decay: `0.01`
- scheduler: `linear`
- warmup steps: `500`
- evaluation strategy: epoch
- save strategy: epoch
- best-model loading enabled
- best metric: `f1`

The example training script overrides batch size to `32` and epochs to `8`.

### `CNNTransformerClassifier`

This is a custom `PreTrainedModel` architecture:

```text
Input IDs
   │
   ▼
Transformer backbone
   │
   ▼
last_hidden_state
   │
   ▼
permute
   │
   ▼
Conv1d(hidden_size → 256, kernel=3, padding=1)
   │
   ▼
ReLU
   │
   ▼
AdaptiveMaxPool1d(1)
   │
   ▼
Linear(256 → num_labels)
   │
   ▼
Logits
```

Training loss is `BCEWithLogitsLoss`.

### `CNNModelTrainer`

Provides a Hugging Face `Trainer` wrapper for `CNNTransformerClassifier`.

Its metric and training configuration largely mirrors `ModelTrainer`, but the supplied implementation does not use the same custom class-weighted loss.

### `CustomTrainer`

Extends Hugging Face `Trainer` and allows an optional externally supplied `compute_loss_fn`.

This is an extension point for custom loss behavior.

---

## 4. Training Pipeline

The supplied `model.py` demonstrates the intended standard training path:

```text
DatasetLoader
    │
    ├── labels
    ├── id2label / label2id
    └── raw dataset
          │
          ▼
AutoTokenizer.from_pretrained(...)
          │
          ▼
DatasetLoader.encode_dataset(...)
          │
          ▼
Model(...)
          │
          ▼
ModelTrainer(...)
          │
          ├── train()
          ├── evaluate()
          └── save_model()
          │
          ▼
model_versions/vN
```

The example script then reloads the saved version and performs inference on a sample message.

---

## 5. Inference Pipeline

The supplied `load_model.py` demonstrates:

```text
DatasetLoader
      │
      ├── label definitions
      └── label mappings
             │
             ▼
      AutoTokenizer
             │
             ▼
      Model(model_version)
             │
             ▼
      input text
             │
             ▼
      tokenizer
             │
             ▼
      model logits
             │
             ▼
      sigmoid probabilities
             │
             ▼
      threshold + argmax
             │
             ▼
      label names
```

The inference script loads `v5`, obtains the dataset label ordering from `DatasetLoader`, and uses that ordering when converting predictions into names.

---

## 6. Label Semantics and Prediction Rules

The implementation is multi-label: each label is represented as an independent binary output.

The standard inference path:

1. Produces logits.
2. Applies sigmoid.
3. Creates an all-zero prediction array.
4. Sets every probability at or above the threshold to `1`.
5. Forces the highest-probability label to `1`, even if it is below the threshold.
6. Converts the resulting binary vector to label names.
7. If `OK` is present, returns only `OK`.

This means the effective inference policy is not a pure threshold classifier: at least one label is always selected.

`OK` is mutually exclusive with toxicity indicator labels.

---

## 7. Data Representation

The source code assumes:

- One text column, normally `Message`.
- Remaining relevant columns are binary label columns.
- Labels are converted to a matrix shaped approximately as:
  `number of examples × number of labels`.
- Tokenized examples contain `input_ids`, `attention_mask`, and `labels`.

The tokenizer uses padding to maximum length, truncation, and a maximum length of `512`.

---

## 8. Data Preparation Operations

### Text filtering

`DataToolkit.filter_text`:

- lowercases text;
- replaces newlines/carriage returns with spaces;
- removes Romanian diacritics:
  - `ă = a`
  - `î = i`
  - `ș = s`
  - `ț = t`
  - `â = a`
- optionally applies supplied regex patterns;
- strips leading whitespace.

`filter_data` additionally removes URLs, HTML-like tags, and characters outside `[0-9a-zA-Z -]`.

### Deduplication

`remove_duplicates` removes duplicate values based on a specified column and keeps the first occurrence.

### Label filtering

`remove_labels_rows` removes rows where any selected label equals a supplied value.

`focus_labels_rows` retains rows where any selected label equals a supplied value.

### Augmentation

Two augmentation paths are present:

- back-to-back translation;
- `nlpaug.RandomWordAug`.

### Resampling

Three approaches are present:

- `RandomOverSampler`;
- `SMOTETomek` over TF-IDF representations;
- `ADASYN` over TF-IDF representations.

These methods encode multi-label vectors as string keys such as `"01010000"` for resampling.

---

## 9. Model Versioning

The supplied scripts use:

```text
./automod-model/model_versions/v4
./automod-model/model_versions/v5
```

A model version appears to contain the files required by Hugging Face `from_pretrained`, with the tokenizer loaded separately from the same version directory.

---

## 11. External Boundaries

- Hugging Face Hub for datasets/models.
- Hugging Face Transformers for tokenization, model loading, and training.
- Hugging Face Datasets for dataset storage/transformation.
- PyTorch for model execution.
- scikit-learn for metrics and TF-IDF.
- imbalanced-learn for resampling.
- `nlpaug` for text augmentation.
- `deep_translator` is referenced by the documented translation workflow.
- Weights & Biases is explicitly disabled in `model.py`.