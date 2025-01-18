from transformers import AutoTokenizer
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from optoolkit import Model, DatasetLoader, Evaluator

if __name__ == '__main__':
    dataset = DatasetLoader(["Message"], "Message")
    tokenizer = AutoTokenizer.from_pretrained('./automod-model/model_versions/v1')
    loaded_dataset = dataset.load_dataset()
    X_test, y_test = dataset.Xy_test('Message', loaded_dataset)

    model_v1 = Model(
            model_name='./automod-model/model_versions/v1',
            num_labels=len(dataset.labels),
            id2label=dataset.id2label,
            label2id=dataset.label2id,
        )
    eval = Evaluator(X_test, y_test, tokenizer)

    eval.evaluate_print(model_v1, avg="micro")
    