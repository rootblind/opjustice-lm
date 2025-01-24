import numpy as np
from transformers import AutoModelForSequenceClassification, EvalPrediction, Trainer, TrainingArguments
from datasets import load_dataset, load_from_disk
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score, multilabel_confusion_matrix
import torch
import nlpaug.augmenter.word as naw
import regex as re
import pandas as pd
import warnings
import os
from math import floor

warnings.filterwarnings("ignore", category=UserWarning)


class DatasetLoader():
    def __init__(self, ignore_columns, text_column, dataset_path='./automod-model/dataset', dataset_name='rootblind/opjustice-dataset'):
        """
            Dataset loader for my huggingface dataset
            - ignore_columns: used for text column, the other columns must be the labels
            - text_column: the text column
            - dataset_path: the path to the dataset locally
            - dataset_name: the name of the dataset on huggingface
        """
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.ignore_columns = ignore_columns
        self.text_column = text_column
        self.dataset = self.load_dataset()
        self.labels = self.get_labels()
        self.id2label, self.label2id = self.create_label_mappings()
        

    def load_dataset(self):
        """
            Loads the dataset and stores it locally

            - returns: dataset
        """
        if os.path.exists(self.dataset_path):
            dataset = load_from_disk(self.dataset_path)
        else:
            os.mkdir(self.dataset_path)
            dataset = load_dataset(self.dataset_name, data_files={"data": "data.csv", "train": "train.csv", "test": "test.csv"})
            dataset.save_to_disk(self.dataset_path)
        return dataset

    def get_labels(self):
        """
            returns: Labels (text column not included)
        """
        return [label for label in self.dataset['train'].features.keys() if label not in self.ignore_columns]

    def create_label_mappings(self):
        """
            Labels are assigned numbers and then numbers are assigned labels
            - returns: dictionaries as explained above
        """
        id2label = {idx: label for idx, label in enumerate(self.labels)}
        label2id = {label: idx for idx, label in enumerate(self.labels)}
        return id2label, label2id

    def preprocess_data(self, examples, tokenizer):
        """
            Preprocesses the text data by tokenizing and encoding it into the appropriate format.
            
            This function takes raw text data from the provided `examples` and uses the tokenizer
            to convert the text into token IDs, along with preparing a multi-label matrix for the labels.
            
            Args:
                examples (dict): A dictionary where the keys are column names (including text and labels) and the values are the data.
                tokenizer: A tokenizer (likely from Hugging Face) used to tokenize the text data.
            
            Returns:
                dict: A dictionary containing tokenized text (e.g., 'input_ids', 'attention_mask') and corresponding labels (multi-label format).
        """
        text = examples[self.text_column]
        encoding = tokenizer(text, padding="max_length", truncation=True, max_length=512)
        labels_batch = {k: examples[k] for k in examples.keys() if k in self.labels}
        labels_matrix = np.zeros((len(text), len(self.labels)))

        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]

        encoding["labels"] = labels_matrix.tolist()
        return encoding

    def encode_dataset(self, tokenizer):
        """
            Applies the preprocess_data function to the entire dataset, encoding the text and labels.
            
            This function uses the preprocess_data function to encode the dataset (including text and labels)
            and formats it into a format suitable for model training, such as PyTorch tensors.
            
            Args:
                tokenizer: A tokenizer (likely from Hugging Face) used to preprocess the text data.
            
            Returns:
                Dataset: The processed dataset in a format suitable for training (e.g., torch tensors).
        """
        encoded_dataset = self.dataset.map(lambda x: self.preprocess_data(x, tokenizer), batched=True, remove_columns=self.dataset['train'].column_names)
        encoded_dataset.set_format("torch")
        return encoded_dataset
    
    def Xy_test(self, text_column, loaded_dataset):
        """
            Fetching the test split of the dataset
        """
        X_test = [sample[text_column] for sample in loaded_dataset['test']]
        y_test = [[sample[label] for label in self.labels] for sample in loaded_dataset['test']]
        return X_test, y_test

class Model:
    def __init__(self, model_name, num_labels, id2label, label2id, device=None):
        """
            This class loads the model and uses its functionalities.
            Built for my personal model, different models might need changes to this code.
            - model_name: the name of the model on huggingface or its local directory path
            - num_labels: the number of labels it has (the dataset it was trained on)
            - id2label and label2id: the dictionaries built using DatasetLoader.create_label_mapping()
            - device: load the model on cuda or cpu
        """
        self.model_name = model_name
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(num_labels, id2label, label2id)

    def load_model(self, num_labels, id2label, label2id):
        """
            Loads the model using huggingface.

            - returns: the model
        """
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name, 
                                                                   problem_type="multi_label_classification", 
                                                                   num_labels=num_labels,
                                                                   id2label=id2label,
                                                                   label2id=label2id).to(self.device)
        
        return model

    def predict(self, text, tokenizer):
        """
            Loads the model to label the text input given.
            The text is encoded and then calculations are done using Sigmoid
            - text: the string input
            - tokenizer: the huggingface tokenizer
            - threshold: Default 0.5 is used to set which labels count

            - returns: The probabilities
        """
        encoding = tokenizer(text, return_tensors="pt")
        encoding = {k: v.to(self.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        logits = outputs.logits
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(logits.squeeze().cpu())
        return probs
    def predictions(self, text, tokenizer, threshold=0.5):
        """
            Converts probabilities into scores
            - returns an array of binary scores depending on probability and threshold
        """
        probs = self.predict(text, tokenizer)
        predictions = np.zeros(probs.shape)
        predictions[np.where(probs >= threshold)] = 1
        predictions[np.argmax(probs)] = 1
        return predictions
    def label_text(self, text, tokenizer, labels, threshold=0.5):
        """
            Converts the Model.predict function's logits into label names
            - text: string as input
            - tokenizer: Model's tokenizer
            - labels: Array of labels
            - threshold: Default 0.5 parameter required by Model.predict 

            - returns: An array that contains only labels that had a probability above threshold
        """
        predictions = self.predictions(text, tokenizer, threshold)
        predicted_labels = [labels[idx] for idx, label in enumerate(predictions) if label]
        if 'OK' in predicted_labels:
            predicted_labels = ['OK']

        return predicted_labels
    def label_dataset(self, dataset, text_column, label_columns, tokenizer, debug=False, threshold=0.5):
        """
            Applies Model.label_text to the entire dataset.
            - dataset: The DataFrame
            - text_column: The column with the raw text
            - label_columns: The columns that the model will assign
            - tokenizer: Model's tokenizer
            - debug: Default False prints in console which text is currently labeled
            - threshold: Default 0.5 parameter required for label_text and predict

            - returns: Labeled DataFrame
        """
        scores = {text_column: []}
        for l in label_columns:
            scores[l] = []
        messages = []
        for _, row in dataset.iterrows():
            
            labels = self.label_text(row[text_column], tokenizer, label_columns, threshold)
            if len(labels) == 0:
                continue
            if debug:
                print(f"Labeling message number {len(messages) + 1}")
            for key in scores:
                if key in labels:
                    scores[key].append(1)
                else:
                    scores[key].append(0)
            messages.append(row[text_column])
        scores[text_column] = messages
        return pd.DataFrame(data=scores)
    
class Evaluator():
    def __init__(self, X_test, y_test, tokenizer):
        """
            Evaluates models loaded through Model
            - X_test: The text column of the test split
            - y_test: The label scores of the test split
            - tokenizer: The Model's tokenizer

            X_test and y_test can be obtained from DatasetLoader.Xy_test()
        """
        self.X_test = X_test
        self.tokenizer = tokenizer
        self.y_test = y_test
    
    def y_pred(self, model):
        """
            Returns the predictions that the model makes on the test split
            - model: The model loaded through Model
        """
        y_pred = [model.predictions(text, self.tokenizer) for text in self.X_test]
        return y_pred
    def evaluate(self, model, avg):
        """
            Evaluates the model based on f1, precision, recall, accuracy and roc_auc scores.
            - model: The model loaded through Model
            - avg: The average method used when scoring the model (micro, macro, samples, weighted)

            - returns: f1 score, precision score, recall score, accuracy score and roc_auc score
        """
        y_pred = self.y_pred(model)
        f1 = f1_score(self.y_test, y_pred, average=avg)
        precision = precision_score(self.y_test, y_pred, average=avg)
        recall = recall_score(self.y_test, y_pred, average=avg)
        accuracy = accuracy_score(self.y_test, y_pred)
        roc_auc = roc_auc_score(self.y_test, y_pred)

        return f1, precision, recall, accuracy, roc_auc
    
    def evaluate_print(self, model, avg):
        """
            Uses the evaluation function to print the results in the console.
            Takes the same parameters.
        """
        f1, precision, recall, accuracy, roc_auc = self.evaluate(model, avg)
        print(f"F1 Score: {f1}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")
        print(f"ROC AUC: {roc_auc}")
        
    def confusion_matrix(self, model):
        """
            The confusion matrix of the multilabel model.
        """
        return multilabel_confusion_matrix(self.y_test, self.y_pred(model))
   
class DataToolkit():
    """
        Multiple functions that modify or generate the data for a dataset. 
    """
    def translate_text(self, translator, text, debug=False):
        """
            Using GoogleTranslate from deep_translator to translate the text input.
            Example usage:

            from deep_translator import GoogleTranslate
            translator = GoogleTranslate("en", "ro")

            translate_text(translator, "Hello") # translates from english to romanian

            - debug: Default False if true, prints the text that is currently being translated

            - returns: the translated string
        """
        if debug:
            print(f'Translated {text[:20]}...')
        return translator.translate(text)
    
    def translate_data(self, translator, data, text_header, debug=False):
        """
            Applies translate_text to the whole dataset.
            Takes ~5m per 1000 messages or 3-4 messages per second

            - translator: the translator to be used
            - data: The dataframe
            - text_header: The text column
            - debug: Default False if true, prints the text that is currently being translated

            - returns: the modified dataframe
        """
        data[text_header] = data[text_header].apply(lambda text: self.translate_text(translator, text, debug))
        return data
    def remove_duplicates(self, data, column):
        """
            Returns the dataframe without duplicates found on the specified column.
            - data: The dataframe
            - column: The column to be checked
        """
        return data.drop_duplicates(subset=column)
    def remove_labels_rows(self, data, labels, value):
        """
            Removes the rows from the dataset that have labels (columns) values equal to the specified value
            - data: The dataframe
            - labels: the columns to be checked
            - value: 0 or 1

            - returns: the modified dataframe
        """
        return data.drop(data[data[labels].eq(value).any(axis=1)].index)
    
    def focus_labels_rows(self, data, labels, value):
        """
            Keeps only the rows that have labels equal to the specified value
            - data: The dataframe
            - labels: the columns to be checked
            - value: 0 or 1

            -return: the modified dataframe
        """
        return data[data[labels].eq(value).any(axis=1)]
    
    def filter_text(self, text, patterns=None):
        """
                Filters the text input given by making it lowercase, swapping diactritics to their counterparts and removes
            the patterns.

            - text: the text to be filtered
            - patterns: the patterns to be removed

            - returns: the filtered text
        """
        text = text.lower()
        if len(text) < 3:
            return text
        text = text.replace('\n', ' ').replace('\r', ' ')
        text = text.replace('ă', 'a')
        text = text.replace('î', 'i')
        text = text.replace('ș', 's')
        text = text.replace('ț', 't')
        text = text.replace('â', 'a')

        if patterns:
            for pattern in patterns:
                text = re.sub(pattern, '', text)
        
        text = text.lstrip()
        return text
    def filter_data(self, data, text_column, labels):
        """
            Uses filter_text to filter the dataframe. Some rows get curated, others are removed
            - data: the dataframe
            - text_column: the column of raw text
            - labels: the labels of the dataset

            - returns: the filtered dataframe
        """
        patterns = [
            re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            re.compile(r'<:(\d+):>'),
            re.compile(r'[^a-zA-Z -]'),
        ]

        filtered_data = {text_column: []}
        for column in labels:
            filtered_data[column] = []

        for _, row in data.iterrows():
            text = self.filter_text(row[text_column], patterns)
            if ' ' not in text or len(text) > 512:
                continue

            if len(text) > 2:
                row[text_column] = text
                for column in labels:
                    filtered_data[column].append(row[column])
                filtered_data[text_column].append(row[text_column])
        return pd.DataFrame(data=filtered_data)
    
    def shuffle(self, df):
        """
            Shuffles the dataframe and returns it.
        """
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    def back_to_back_translation(self, data, text_column, translator_primary, translator_secondary, debug=False):
        """
            Back to back translation is an augmentation method.
            Uses translate_data to translate the dataframe into the specified translator language and then translates it back.

            - data: the dataframe to translate
            - text_column: the column of text
            - translator_primary: the translator that translates the dataframe from original language to the specified one
            - translator_secondary: the translator that translates the dataframe from the secondary language to the original language
            - debug: Default False if true, prints the text that is currently being translated

            - returns: the augmented dataframe

            primary: from primary language to secondary
            secondary: from secondary language to primary language

            primary language -> translate to secondary language -> translate back to primary language
        """
        
        data = self.translate_data(translator_primary, data, text_column, debug)
        data = self.translate_data(translator_secondary, data, text_column, debug)
        return data # must be assigned to the entire dataset, not only the text column
    def aug_naw(self, text, aug):
        """
            Using nlpaug.RandomWordAug to augment the given text.
            - text: the text to augment
            - aug: naw.RandomWordAug object is expected
        """
        return aug.augment(text)[0]
    
    def data_aug_naw(self, data, text_column, action, aug_min=1, aug_max=None, stopwords=None,
                     stopwords_regex=None, target_words=None, case_sensitive=True, verbose=0):
        """
            Augments the dataset using naw.RandomWordAug
            - data: the dataframe
            - text_column: the column of text
            - action: The action used when performing the augmentation (swap, substitute, delete)
            Other arguments are used when initializing the augmentation object.

            - returns: the augmented dataframe
        """
        aug = naw.RandomWordAug(action=action,
                                aug_min=aug_min,
                                aug_max=aug_max,
                                stopwords=stopwords,
                                stopwords_regex=stopwords_regex,
                                target_words=target_words,
                                case_sensitive=case_sensitive,
                                verbose=verbose
                                )
        data[text_column] = data[text_column].apply(lambda text: self.aug_naw(text, aug))
        return data


    def split_train_test(self, data, labels, split_size=0.2):
        """
        Splits the DataFrame into training and testing sets while maintaining label proportions.

        Parameters:
        - data: pd.DataFrame, input data containing the text and label columns.
        - labels: list of str, columns corresponding to labels.
        - split_size: float, proportion of the dataset to include in the test split.

        Returns:
        - train: pd.DataFrame, training subset.
        - test: pd.DataFrame, testing subset.
        """
        # Separate rows into categories based on labels
        category_messages = {label: data[data[label] == 1] for label in labels}
        
        # Shuffle rows for each category
        for label in labels:
            category_messages[label] = category_messages[label].sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Calculate split indices
        split_index = {label: floor(len(category_messages[label]) * split_size) for label in labels}
        
        # Split data into train and test sets
        train_frames = []
        test_frames = []
        for label in labels:
            test_frames.append(category_messages[label].iloc[:split_index[label]])
            train_frames.append(category_messages[label].iloc[split_index[label]:])
        
        # Combine all categories into single DataFrames
        train = pd.concat(train_frames).sample(frac=1, random_state=42).reset_index(drop=True)
        test = pd.concat(test_frames).sample(frac=1, random_state=42).reset_index(drop=True)
        
        return train, test



    
class RegexClassifier():
    def __init__(self, triggerWords, max_typo=1):
        """
            Generates regex patterns and checks if given text respects the patterns.
            - triggerWords: array of strings used to generate the regex
            - max_typo: Default 1 The regex patterns can be made to admit a degree of typos when searching for the pattern.
        """
        self.max_typo = max_typo
        self.triggerWords = triggerWords

    def regex_gen(self, word):
        """
            Using regex library to generate the pattern for given word.

            - returns: pattern string
        """
        max_typo = self.max_typo
        escaped_word = re.escape(word)

        pattern = f"({escaped_word}){{e<={max_typo}}}"
        return pattern
    
    def trigger_patterns(self, triggerWords):
        """
            Using regex_gen to create a string of patterns for the given array of strings.

            - returns: string of patterns
        """
        triggerPatterns = []
        for word in triggerWords:
            triggerPatterns.append(self.regex_gen(word))
        
        return  "|".join(triggerPatterns)
    
    def regex_classifier(self, message):
        """
            Returns a boolean of weather the patterns match the message
        """
        triggerWords = self.triggerWords
        triggerPattern = self.trigger_patterns(triggerWords)
        return bool(re.search(triggerPattern, message.lower()))
    
class DataRegexClassifier():
    def __init__(self, classifier, text_header, label):
        """
            Corrects a label of the dataframe using RegexClassifier patterns
            - classifier: RegexClassifier object
            - text_header: The column of the messages
            - label: the label to be associated with the values
        """
        self.text_header = text_header
        self.label = label
        self.classifier = classifier
        
    def classify_data(self, df):
        """
            Takes a dataframe and adds a new label. Due to the structure of this class, this classifier is single class.

            - returns: the dataframe
        """
        label = self.label
        text_header = self.text_header
        classifier = self.classifier
        df[label] = df[text_header].apply(lambda message: 1 if classifier.regex_classifier(message) else 0)

        return df
    

class ModelTrainer:
    def __init__(self, model, tokenizer, train_dataset, eval_dataset, output_dir, batch_size=16, metric_name="f1", epochs=8,
                 save_steps=500, weight_decay=0.01, logging_steps=10, lr_scheduler_type='linear', warmup_steps=500):
        """
        Initializes the Trainer class with the provided configuration parameters.

        - model (nn.Module): The model to be trained or evaluated.
        - tokenizer (PreTrainedTokenizer): The tokenizer for encoding input text.
        - train_dataset (Dataset): The training dataset.
        - eval_dataset (Dataset): The evaluation dataset.
        - output_dir (str): The directory to save model checkpoints and results.
        - batch_size (int, default=16): The batch size for training and evaluation.
        - metric_name (str, default="f1"): The evaluation metric to track.
        - epochs (int, default=8): The number of training epochs.
        - save_steps (int, default=500): The number of steps between model saves.
        - weight_decay (float, default=0.01): Weight decay for regularization.
        - logging_steps (int, default=10): Number of steps between logging outputs.
        - lr_scheduler_type (str, default='linear'): Type of learning rate scheduler.
        - warmup_steps (int, default=500): Number of warmup steps for learning rate scheduler.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.metric_name = metric_name
        self.epochs = epochs
        self.save_steps = save_steps
        self.weight_decay = weight_decay
        self.logging_steps = logging_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.warmup_steps = warmup_steps
        self.trainer = self.initialize_trainer()

    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        """
        Calculates multi-label classification metrics including F1 score, ROC AUC, and accuracy.

        - predictions (np.ndarray): The predicted labels or probabilities.
        - labels (np.ndarray): The ground truth labels.
        - threshold (float, default=0.5): Threshold to classify predictions as positive or negative.

        - returns (dict): A dictionary containing 'f1', 'roc_auc', and 'accuracy' metrics.
        """
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        y_true = labels
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_pred, average='micro')
        accuracy = accuracy_score(y_true, y_pred)
        metrics = {'f1': f1_micro_average, 'roc_auc': roc_auc, 'accuracy': accuracy}
        return metrics

    def compute_metrics(self, p: EvalPrediction):
        """
        Computes evaluation metrics using multi_label_metrics.

        - p (EvalPrediction): An object containing predictions and labels for evaluation.

        - returns (dict): The metrics computed using multi_label_metrics.
        """
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        return self.multi_label_metrics(preds, p.label_ids)

    def initialize_trainer(self):
        """
        Initializes the Hugging Face Trainer with the provided configuration parameters.

        - returns (Trainer): A Hugging Face Trainer instance ready for training and evaluation.
        """
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            load_best_model_at_end=True,
            metric_for_best_model=self.metric_name,
            save_steps=self.save_steps,
            weight_decay=self.weight_decay,
            logging_steps=self.logging_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            warmup_steps=self.warmup_steps
        )

        return Trainer(
            self.model,
            training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        """
        Starts the training process using the initialized Trainer.

        - returns (None): The function performs training without returning anything.
        """
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        self.trainer.train()

    def evaluate(self):
        """
        Evaluates the model on the evaluation dataset.

        - returns (dict): The evaluation results, typically containing loss and metrics like accuracy and F1 score.
        """
        return self.trainer.evaluate()

    def save_model(self):
        """
        Saves the trained model to the specified output directory.

        - returns (None): The function saves the model without returning anything.
        """
        for param in self.model.parameters():
            if not param.is_contiguous():
                param.data = param.data.contiguous()
        self.trainer.save_model(self.output_dir)